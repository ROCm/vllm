```
docker build -t api-server-olga .  

docker run --rm -dt -p 9000:9000 -v $HOME/:/mnt/home/ --name api-server-olga api-server-olga 
```