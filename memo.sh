# 同期コマンド
# Local → Server
rsync -avz --progress /Users/kazuki-h/research/multiproject_research/ socsel:/mnt/data1/kazuki-h/multiproject_research/


# Server → Local
rsync socsel:/mnt/data1/kazuki-h/multiproject_research/ /Users/kazuki-h/research/multiproject_research/