class Agent:
    def __init__(self):
        pass 

if __name__ == "__main__":
    agent = Agent()
    data = data.get_data_signals()
    new_squad = algorithm.get_new_squad(data)
    agent.submit()