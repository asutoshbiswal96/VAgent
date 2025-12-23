from src.agent import LICAgent
import os

if __name__ == '__main__':
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, 'data', 'policyholders.csv')
    csv_path = os.path.abspath(csv_path)
    agent = LICAgent(csv_path)
    print('Available policy ids: P001..P005')
    pid = input('Choose policy id: ').strip()
    agent.start_conversation(pid)
