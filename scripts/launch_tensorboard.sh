sudo docker start rooftopml
sudo docker exec -it rooftopml tensorboard --host 0.0.0.0 --logdir ./training_logs/$1
