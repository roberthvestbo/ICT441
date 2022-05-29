# Install needed libraries
# pip install gym_super_mario_bros==7.3.0 nes_py
# pip install stable-baselines3[extra]
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Import the game
import gym
import gym_super_mario_bros
# Import the joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Import frame stack and grayscaling wrapper
from gym.wrappers import GrayScaleObservation
# Import vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecVideoRecorder
# Import Matplotlib to show the impact of stack framing
from matplotlib import pyplot as plt
# Import os for file path management
import os
# Import PPO for algos
from stable_baselines3 import PPO
# Import base callback for saving models
from stable_baselines3.common.callbacks import BaseCallback


class CustomRewardGoal2A(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardGoal2A, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info["score"] - self._current_score) / 40.0
        self._current_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info


class CustomRewardGoal2B(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardGoal2B, self).__init__(env)
        self._current_score = 0
        self._number_of_lives = 2

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward = (info["score"] - self._current_score) / 10

        if info["life"] == 255:
            life_loss = True
        else:
            life_loss = (self._number_of_lives - info["life"]) > 0

        self._current_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 350.0
            else:
                reward -= 100.0
                self._number_of_lives = 2
                self._current_score = 0

        if life_loss and not done:
            reward -= 50.0
            self._number_of_lives = info["life"]

        print(reward)
        return state, reward, done, info


class CustomRewardGoal2C(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardGoal2C, self).__init__(env)
        self._current_score = 0
        self._number_of_lives = 2
        self._current_time = 400

        # starting point
        self._current_x_pos = 40
        self._max_x_pos_memory = 0
        self._previous_x_pos_memory = 0
        self._steps_run_wrong_direction = 0

    def step(self, action):
        def reset():
            self._current_x_pos = 40
            self._current_time = 400
            self._steps_run_wrong_direction = 0
            self._number_of_lives = 2

        state, reward, done, info = self.env.step(action)
        score_diff = info["score"] - self._current_score
        life_loss = (self._number_of_lives - info["life"]) > 0

        # clip to avoid too high reward for mushroom/flower (1000p) and coins (200p)
        reward += min(score_diff, 150)

        time_diff = self._current_time - info["time"]

        if self._max_x_pos_memory < info["x_pos"]:
            self._max_x_pos_memory = info["x_pos"]
        else:
            self._steps_run_wrong_direction += time_diff

        #  Handle when the agent hits the left wall
        standstill = self._previous_x_pos_memory == info["x_pos"]
        if standstill:
            self._steps_run_wrong_direction += time_diff

        self._previous_x_pos_memory = info["x_pos"]

        # reward movement also for left direction /normal reward for left is -3
        # Make sure to force progress after moving in the wrong direction too long.
        if info["x_pos"] < self._current_x_pos:
            reward -= min((self._steps_run_wrong_direction / 100), 10)
        elif standstill:
            reward -= min((self._steps_run_wrong_direction / 100), 10)
        """elif score_diff > 1:
            self._steps_run_wrong_direction = 0"""

        self._current_score = info["score"]
        self._current_x_pos = info["x_pos"]
        self._current_time = info["time"]

        if done:
            if info["flag_get"]:
                reward += 350.0
            else:
                reward -= 100.0
                reset()

        if life_loss:
            reward -= 50.0
            self._number_of_lives = info["life"]
            reset()

        # print(reward / 10)
        return state, reward / 10, done, info


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


def show_image_stack():
    print(SIMPLE_MOVEMENT[env.action_space.sample()])
    env.reset()

    # Make steps to fill the
    env.step([env.action_space.sample()])
    state, reward, done, info = env.step([env.action_space.sample()])

    plt.figure(figsize=(10, 8))
    for idx in range(state.shape[3]):
        plt.subplot(1, 4, idx + 1)
        plt.imshow(state[0][:, :, idx])
    plt.show()


def train_model(callback_dir, log_dir, resume):
    # Setup model saving callback
    callback = TrainAndLoggingCallback(check_freq=50000, save_path=callback_dir)

    learning_rate = 0.0001

    # This is the AI model started
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir, learning_rate=learning_rate, n_steps=512)

    if resume:
        model = PPO.load('./train/best_model_4500000')
        model.set_env(env)

    # Train the AI model, this is where the AI model starts to learn
    model.learn(total_timesteps=6000000, callback=callback)


def run_the_game(load_model):
    if load_model:
        # Load model
        model = PPO.load('./right-and-left-10/best_model_300000')
    else:
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=512)

    # Start the game
    state = env.reset()
    # Loop through the game
    while True:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()


def record_video():
    for i in range(200):
        # Create the base environment
        env = gym_super_mario_bros.make("SuperMarioBros-v0")
        # My custom reward function
        # env = CustomReward(env)
        env = CustomRewardGoal2C(env)
        # Simplify the controls
        # I use another JoypadSpace to allow moving left
        # customMovement = [['right', 'B'], ['right', 'A', 'B'], ['A'], ['left', 'B'], ['left', 'A', 'B']]
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        # print(SIMPLE_MOVEMENT)
        # Grayscale
        env = GrayScaleObservation(env, keep_dim=True)
        # Wrap inside the Dummy Environment
        env = DummyVecEnv([lambda: env])
        # Stack the frames
        env = VecFrameStack(env, 4, channels_order="last")

        env_id = "smb"
        video_folder = 'logs/videos'
        video_length = 10000
        name_prefix = "reward8-" + str(i)
        max_score = 0

        model = PPO.load('./train/best_model_4500000')

        envv = VecVideoRecorder(env, video_folder,
                                record_video_trigger=lambda x: x == 0, video_length=video_length,
                                name_prefix=name_prefix.format(env_id))

        state = envv.reset()

        for _ in range(video_length + 1):
            action, _ = model.predict(state)
            state, reward, done, info = envv.step(action)

            score = info[0].get("score")
            if score > max_score:
                max_score = score

        print(name_prefix + "_" + str(max_score))

        # Save the video
        envv.close_video_recorder()


# Create the base environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
# My custom reward function
# env = CustomReward(env)
env = CustomRewardGoal2C(env)
# Simplify the controls
customMovement = [['right', 'B'], ['right', 'A', 'B'], ['A'], ['left', 'B'], ['left', 'A', 'B']]
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# print(SIMPLE_MOVEMENT)
# Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# Stack the frames
env = VecFrameStack(env, 4, channels_order="last")

"""Preview the image stack"""
# show_image_stack()

"""Train Model"""
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/train/'
# train_model(CHECKPOINT_DIR, LOG_DIR, False)


"""RUNNING THE GAME FROM TRAINED MODEL"""
run_the_game(True)
# record_video()

# print(tensorflow.config.list_physical_devices())
