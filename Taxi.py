import gym
import time

# تعریف نگاشت اقدامات به جهت‌ها
actions = {
    0: "South",  # پایین
    1: "North",  # بالا
    2: "East",  # راست
    3: "West",  # چپ
    4: "Pickup",  # سوار کردن مسافر
    5: "Dropoff"  # پیاده کردن مسافر
}

env = gym.make('Taxi-v3', render_mode='ansi')  # یا render_mode='human' برای نمایش در پنجره

env.reset()
print(env.render())

episode_num = 100
for i in range(episode_num):
    print(f"Episode {i}")

    # در صورتی که env.reset() یک tuple برگرداند، فقط state را بگیرید
    initial_state = env.reset()
    state = initial_state if isinstance(initial_state, int) else initial_state[0]

    # استخراج اطلاعات اولیه از state
    taxi_row, taxi_col, passenger_loc, destination = env.decode(state)

    while True:
        action = env.action_space.sample()
        next_state, reward, done, _, info = env.step(action)

        # اگر next_state به صورت tuple است، فقط مقدار اول را بگیرید
        next_state = next_state if isinstance(next_state, int) else next_state[0]

        # نمایش اقدام به همراه جهت، موقعیت مسافر، پاداش و وضعیت نهایی
        action_name = actions[action]

        # استخراج موقعیت‌های جدید
        taxi_row, taxi_col, passenger_loc, destination = env.decode(next_state)
        passenger_position = "in Taxi" if passenger_loc == 4 else f"at {['R', 'G', 'Y', 'B'][passenger_loc]}"
        destination_position = f"at {['R', 'G', 'Y', 'B'][destination]}"

        print(
            f"Current state: Taxi at ({taxi_row}, {taxi_col}), Passenger {passenger_position}, Destination {destination_position}")
        print(f"Action: {action_name} -> Reward: {reward} -> Done: {done}")
        print(env.render())
        time.sleep(0.5)

        state = next_state
        if done:
            print(f"Episode ended with reward: {reward}\n")
            break

env.close()
