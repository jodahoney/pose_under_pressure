import gymnasium as gym


def main():
    env = gym.make("Walker2d-v5", render_mode=None)
    obs, info = env.reset(seed=0)

    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"t={t}, reward={reward:.3f}, "
            f"terminated={terminated}, truncated={truncated}"
        )

    env.close()


if __name__ == "__main__":
    main()
