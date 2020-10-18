import pandas as pd

def get_team_feature(id, feature):
    a = pd.read_csv("./data/2019-20/teams.csv")
    b = pd.read_csv("./data/2020-21/teams.csv")
    teams = pd.merge(b, a, on=['id'], how='outer', suffixes=("", "_prev"))
    team = teams[teams["id"] == id]["name"].values[0]
    team_file_name = team.replace(" ", "_")
    team_history = pd.read_csv(f"./data/2019-20/understat/understat_{team}.csv")
    team_current = pd.read_csv(f"./data/2020-21/understat/understat_{team}.csv")
    team_history = team_history.reset_index()
    team_history["id"] = id
    team_history["index"] += 1
    team_current = team_current.reset_index()
    team_current["id"] = id
    team_current["index"] += 53
    team_history = pd.concat((team_history, team_current))
    team_history = pd.pivot(index="id", columns="index", values=feature, data=team_history)
    return team_history

if __name__ == "__main__":
    print(get_team_feature(1, "npxG"))