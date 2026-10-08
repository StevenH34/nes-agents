"""Super Mario Bros. (NES, Mapper 0). Checked against the real ROM: boot lands in 1-1, running right earns
progress reward, the timer penalty fires every 24 frames, and running into the first Goomba ends the episode.

RAM addresses and the reward follow gym-super-mario-bros (https://github.com/Kautenja/gym-super-mario-bros) and the
SMB RAM map on Data Crystal. The reward is rightward progress, minus a penalty for each in-game clock tick and on
death, plus a bonus for reaching the flagpole, clipped to [-15, 15] per step. An episode is one life.
"""

import nes_py
import numpy as np

from nes_gym.games.base import GameSpec

A = nes_py.BUTTON_A
B = nes_py.BUTTON_B
START = nes_py.BUTTON_START
LEFT = nes_py.BUTTON_LEFT
RIGHT = nes_py.BUTTON_RIGHT

# RAM addresses.
PLAYER_STATE = 0x000E  # 0x06 = dead, 0x08 = normal play, 0x0B = dying
FLOAT_STATE = 0x001D  # 0x03 = sliding down the flagpole
X_PAGE = 0x006D  # x-position = page * 256 + x within the page
X_IN_PAGE = 0x0086
Y_VIEWPORT = 0x00B5  # > 1 means below the bottom of the screen (fell into a pit)
LEVEL = 0x075C  # 0-based
LIVES = 0x075A
WORLD = 0x075F  # 0-based
GAME_MODE = 0x0770  # 0 = title screen / demo, 1 = playing
PAUSE_STATUS = 0x0776  # bit 0 set while paused; 0x81 on pausing, 0x01 paused, 0x80 unpausing
TIMER_DIGITS = (0x07F8, 0x07F9, 0x07FA)  # one decimal digit per byte, hundreds first

STATE_DEAD = 0x06
STATE_NORMAL = 0x08
STATE_DYING = 0x0B
FLOAT_FLAGPOLE = 0x03
MODE_TITLE = 0
MODE_PLAYING = 1

# A jump in x larger than this in one step is a teleport (pipe, level change), not movement, so it earns nothing.
MAX_X_STEP = 50
DEATH_PENALTY = -25.0
FLAG_BONUS = 50.0
REWARD_CLIP = 15.0

# Boot timing, in frames.
TITLE_WAIT = 60
START_HOLD = 4
BOOT_LIMIT = 2000


def x_position(ram: np.ndarray) -> int:
    return int(ram[X_PAGE]) * 256 + int(ram[X_IN_PAGE])


def timer(ram: np.ndarray) -> int:
    hundreds, tens, ones = (int(ram[a]) for a in TIMER_DIGITS)
    return hundreds * 100 + tens * 10 + ones


def is_dying(ram: np.ndarray) -> bool:
    return int(ram[PLAYER_STATE]) in (STATE_DEAD, STATE_DYING) or int(ram[Y_VIEWPORT]) > 1


def at_flagpole(ram: np.ndarray) -> bool:
    return int(ram[FLOAT_STATE]) == FLOAT_FLAGPOLE


class SmbSpec(GameSpec):
    # NOOP, R, R+A, R+B, R+A+B, A, L
    actions = (0, RIGHT, RIGHT | A, RIGHT | B, RIGHT | A | B, A, LEFT)

    def boot(self, core: nes_py.NesCore) -> None:
        """Power-on → title screen → press Start → wait until the level timer is set (400 in 1-1) and Mario is in
        normal play.

        Start is only pressed while still on the title screen, because pressing it during play pauses the game.
        """
        core.step(0, frames=TITLE_WAIT)
        start_timer = None
        for _ in range(BOOT_LIMIT):
            ram = core.ram()
            if int(ram[GAME_MODE]) == MODE_TITLE:
                core.step(START, frames=START_HOLD)
                core.step(0, frames=START_HOLD)
                continue
            if start_timer is None:
                start_timer = timer(ram)
            elif timer(ram) != start_timer and self.playing(ram):
                return
            core.step(0)
        raise RuntimeError("Super Mario Bros. did not reach gameplay; is this the right ROM?")

    def reward(self, prev_ram: np.ndarray, ram: np.ndarray) -> float:
        dx = x_position(ram) - x_position(prev_ram)
        if abs(dx) > MAX_X_STEP:
            dx = 0
        # The timer only goes down; ignore resets (e.g. a new level) that would make this positive.
        time_penalty = min(timer(ram) - timer(prev_ram), 0)
        death = DEATH_PENALTY if is_dying(ram) else 0.0
        flag = FLAG_BONUS if at_flagpole(ram) and not at_flagpole(prev_ram) else 0.0
        return float(np.clip(dx + time_penalty + death + flag, -REWARD_CLIP, REWARD_CLIP))

    def terminated(self, ram: np.ndarray) -> bool:
        return is_dying(ram) or at_flagpole(ram)

    def playing(self, ram: np.ndarray) -> bool:
        """Normal play: not the title screen (whose attract demo also uses the normal player state), the lives
        screen, the castle walk or a level's walk-in. Both checks are needed: the lives screen is in playing mode."""
        return (
            int(ram[GAME_MODE]) == MODE_PLAYING
            and int(ram[PLAYER_STATE]) == STATE_NORMAL
            and not self.terminated(ram)
        )

    def paused(self, ram: np.ndarray) -> bool:
        """Paused with Start: the game is frozen from the press until it unpauses."""
        return bool(ram[PAUSE_STATUS] & 1)

    def info(self, ram: np.ndarray) -> dict:
        return {
            "x": x_position(ram),
            "world": int(ram[WORLD]) + 1,
            "level": int(ram[LEVEL]) + 1,
            "lives": int(ram[LIVES]),
            "time": timer(ram),
            "flag_get": at_flagpole(ram),
        }
