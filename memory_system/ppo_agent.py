# memory_system/ppo_agent.py
from enum import Enum
import numpy as np
from stable_baselines3 import PPO
from gym import spaces

class ActionStatus(Enum):
    IN_PROGRESS = 0
    ALMOST_DONE = 1
    DONE = 2
    INTERRUPTED = 3

class _EnumLike(str):
    def __new__(cls, value: str):
        obj = str.__new__(cls, value)
        obj.value = value
        return obj

class _ReflectionStub:
    def __init__(self, op="noop"):
        self.vision = []
        self.last_action = _EnumLike(op)                # ActionType
        self.last_action_result = _EnumLike("success")  # ResultType
        self.last_action_result_reflection = ""
        self.last_action_repeated_reflection = ""

class _GoalStub:
    def __init__(self):
        self.ultimate_goal = _EnumLike("collect_diamond")   # LongTermGoalType
        self.long_term_goal = _EnumLike("collect_diamond")  # LongTermGoalType
        self.long_term_goal_subgoals = ""
        self.long_term_goal_progress = _EnumLike("share")   # GoalType
        self.long_term_goal_status = _EnumLike("in_progress")  # ResultType
        self.current_goal = _EnumLike("share")              # GoalType
        self.current_goal_reason = ""
        self.current_goal_status = _EnumLike("in_progress")    # ResultType

class _NextActionStub:
    def __init__(self, op="noop"):
        self.next_action = _EnumLike(op)                      # ActionType
        self.next_action_reason = ""
        self.next_action_prerequisites_status = _EnumLike("success")  # ResultType
        self.next_action_prerequisites = ""
        self.final_next_action = _EnumLike(op)                # ActionType
        self.final_next_action_reason = ""
        self.final_next_action_status = _EnumLike("success")  # ResultType
        self.final_target_material_to_collect = _EnumLike("grass")          # NavigationDestinationItems
        self.final_target_material_to_share = _EnumLike("not_applicable")   # ShareableItems
        self.final_target_agent_id = -1

class _ResponseEventStub:
    def __init__(self, op="noop"):
        self.epsiode_number = 0
        self.timestep = 0
        self.past_events = ""
        self.current_facing_direction = _EnumLike("grass")  # MaterialType
        self.current_inventory = []
        self.collaboration = None
        self.reflection = _ReflectionStub(op)
        self.goal = _GoalStub()
        self.action = _NextActionStub(op)
        self.summary = ""

class _EpisodicMemoryStub:
    def __init__(self, op="noop"):
        self.final_response = _ResponseEventStub(op)
    def generate_summary(self):
        return ""

class _ExperienceStub:
    def __init__(self, op="noop"):
        self.episodic_memory = _EpisodicMemoryStub(op)



class PPOAgent:
    def __init__(
        self,
        id,
        policy_path: str,
        action_names,
        deterministic: bool = True,
        chw_obs_shape=(3, 64, 64),  # CHW for SB3 CNN
        n_actions: int = 17,
        device: str = "auto",
    ):
        self.id = id
        self.action_status = ActionStatus.DONE
        self.op = _EnumLike("noop")
        self.action = _EnumLike("noop")
        self.rss_to_collect = _EnumLike("not_applicable")
        self.rss_to_share = _EnumLike("not_applicable")
        self.target_agent_id = -1
        self.summary = ""
        self.wm_content = ""
        self.think_context = ""
        self.is_human = False
        self.experience = _ExperienceStub(self.op)
        self.replay = []

        self._action_names = action_names
        self._deterministic = deterministic

        obs_space = spaces.Box(low=0, high=255, shape=chw_obs_shape, dtype=np.uint8)
        act_space = spaces.Discrete(n_actions)
        self._model = PPO.load(
            policy_path,
            custom_objects={"observation_space": obs_space, "action_space": act_space},
            device=device,
            print_system_info=False,
        )
        self._last_obs = None

    # Functions mimicking LLM agent behavior
    def update_state(self, obs, step, env, info, episode_number, episode_timestep):
        self._last_obs = obs
        try:
            row = {'step': step}
            if isinstance(info, dict):
                if 'achievements' in info and isinstance(info['achievements'], dict):
                    row.update(info['achievements'])
                if 'inventory' in info and isinstance(info['inventory'], dict):
                    row.update(info['inventory'])
            self.replay.append(row)
        except Exception:
            pass

    def update_action_status(self, status):
        self.action_status = status

    def update_current_skill(self, op, rss_to_collect, rss_to_share, target_agent_id):
        self.op = op
        self.rss_to_collect = _EnumLike(rss_to_collect)
        self.rss_to_share = _EnumLike(rss_to_share)
        self.target_agent_id = target_agent_id

    def update_action(self, action):      # <-- add
        self.action = action

    def create_experience(self):
        return ""

    def think(self, context):
        return None

    def consolidate_experience(self, final_response):
        return self.op, self.rss_to_collect, self.rss_to_share, self.target_agent_id

    # PPO action --------------------------------------------------
    def act(self, obs_rgb_hwc: np.ndarray):
        # Ensure HWC uint8 64x64x3, then convert to CHW for CNN
        arr = np.asarray(obs_rgb_hwc)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] != 3:
            arr = arr[..., :3] if arr.shape[-1] > 3 else np.pad(arr, ((0,0),(0,0),(0,3-arr.shape[-1])), mode="edge")
        if arr.shape[:2] != (64, 64):
            from PIL import Image
            arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((64, 64), Image.BILINEAR))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        chw = np.transpose(arr, (2, 0, 1))

        action_idx, _ = self._model.predict(chw, deterministic=self._deterministic)
        action_idx = int(action_idx)
        self.op = _EnumLike(self._action_names[action_idx])
        self.action = self.op             # <-- keep in sync
        self.action_status = ActionStatus.IN_PROGRESS
        self.experience.episodic_memory.final_response = _ResponseEventStub(self.op)
        return self.op, action_idx
