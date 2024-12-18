This is the first iteration of Buildkite monitoring DB. 

Running the script requires BUILDKITE_API_TOKEN, request it from [here](https://buildkite.com/user/api-access-tokens).  
Create .env file in .buildkite_monitor directory and paste there:
> export BUILDKITE_API_TOKEN='YOUR_TOKEN' 

Running the script also requires having GMAIL_USERNAME and GMAIL_PASSWORD in .env.

Use py_3.9 conda environment for development.

One option for a scheduled run would be to use cron:\
On some machine, edit your crontab file:\
`crontab -e`\
Using nano/vim/emacs to add an entry for running the container, e.g.\
`*/15 * * * * docker run --network=host --rm -v $HOME/:/mnt/home/ --name=hissu-bkmonitor hissu-bkmonitor` # Runs the hissu-bkmonitor container every 15 minutes    

To cancel or change the cron job:  

`crontab -e`\
Comment out or change the line with the job, then  
`service cron reload`




