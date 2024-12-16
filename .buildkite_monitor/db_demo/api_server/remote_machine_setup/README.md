Docker container runs a data_uploader.py script which sends data to ixt-hq-ubb4-33 machine.

Build it:  
`docker build -t agent_health-data-uploader .`


To schedule it edit crontab file:\
`crontab -e`\
Using nano/vim/emacs to add an entry for running the container, e.g.\
`*/15 * * * * docker run --rm --name agent_health-data-uploader agent_health-data-uploader` # Runs the agent_health-data-uploader container every 15 minutes    
