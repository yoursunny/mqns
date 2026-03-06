# extctrl

```bash
docker rm -f nats_server
docker run -d --name nats_server -p 4222:4222 nats:2 -js

wget https://github.com/nats-io/natscli/releases/download/v0.3.1/nats-0.3.1-amd64.deb
sudo apt install ./nats-0.3.1-amd64.deb
```
