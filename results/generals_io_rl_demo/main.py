from generals.agents import RandomAgent, ExpanderAgent
from generals.envs import PettingZooGenerals
from generals import GridFactory

def main():
    # Initialize agents
    random_agent = RandomAgent()
    expander_agent = ExpanderAgent()

    # Names are used for the environment
    agent_names = [random_agent.id, expander_agent.id]
    # Store agents in a dictionary
    agents = {
        random_agent.id: random_agent,
        expander_agent.id: expander_agent
    }

    # Initialize grid factory for consistent map generation
    grid_factory = GridFactory(
        mode="uniform",
        min_grid_dims=(15, 15),
        max_grid_dims=(23, 23),
        mountain_density=0.2,
        city_density=0.05,
        seed=42, # Fixed seed for reproducibility
    )

    # Create environment
    # Changed render_mode to None for headless execution in Docker.
    # For local execution with GUI, change to render_mode="human" and ensure Pygame dependencies are met.
    env = PettingZooGenerals(agents=agent_names, grid_factory=grid_factory, render_mode=None)
    
    print("Starting Generals.io simulation...")
    observations, info = env.reset()
    print(f"Initial observation for {env.agents[0]}: Land={observations[env.agents[0]]['owned_land_count']}, Army={observations[env.agents[0]]['owned_army_count']}")

    terminated = truncated = False
    step_count = 0
    max_steps = 500 # Limit steps to prevent infinite loops in demo

    while not (terminated or truncated) and step_count < max_steps:
        actions = {}
        for agent_id in env.agents:
            # Ask agent for action
            actions[agent_id] = agents[agent_id].act(observations[agent_id])
        
        # All agents perform their actions
        observations, rewards, terminated, truncated, info = env.step(actions)
        # env.render() # Render is disabled for headless Docker execution
        
        step_count += 1
        if step_count % 50 == 0:
            print(f"Step {step_count}:")
            for agent_id in env.agents:
                if agent_id in observations: # Check if agent still active
                    print(f"  {agent_id}: Land={observations[agent_id]['owned_land_count']}, Army={observations[agent_id]['owned_army_count']}, Reward={rewards.get(agent_id, 0)}")
                else:
                    print(f"  {agent_id}: Eliminated")

    print("\nSimulation finished.")
    if terminated:
        print("Game terminated.")
    elif truncated:
        print("Game truncated (max steps reached).")
    else:
        print("Game ended unexpectedly.")

    # Report final rewards/winner
    winner_found = False
    for agent_id in env.agents:
        if rewards.get(agent_id, 0) > 0:
            print(f"Winner: {agent_id} (Reward: {rewards[agent_id]})\n")
            winner_found = True
        elif rewards.get(agent_id, 0) < 0:
            print(f"Loser: {agent_id} (Reward: {rewards[agent_id]})\n")
    if not winner_found:
        print("No clear winner reported (possibly truncated or draw).\n")

    env.close()

if __name__ == "__main__":
    main()