import pandas as pd
from dotenv import load_dotenv
import numpy as np
from pandas import Timestamp

import os
import json
import requests
from datetime import datetime
import zoneinfo
import warnings
import glob
import difflib
import time


import db_functions 
from importlib import reload
reload(db_functions)
 
from db_functions import fetch_data, types_fix, bronze_builds, bronze_jobs_logs, bronze_agents, calculate_wait_time


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
warnings.filterwarnings('ignore')

load_dotenv()

BUILDKITE_API_TOKEN = os.getenv('BUILDKITE_API_TOKEN')
GMAIL_USERNAME = os.getenv('GMAIL_USERNAME')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
ORGANIZATION_SLUG='vllm'
PIPELINE_SLUG = 'ci'#'ci-aws'
LAST_24_HOURS = (datetime.utcnow() - pd.Timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M')
WAITING_TIME_ALERT_THR = 10800 # 3 hours
AGENT_FAILED_BUILDS_THR = 3 # agents declaired unhealthy if they have failed jobs from >=3 unique builds
RECIPIENTS = ['hissu.hyvarinen@amd.com', 'olga.miroshnichenko@amd.com', 'alexei.ivanov@amd.com']
PATH_TO_LOGS = '/mnt/home/buildkite_logs/'
PATH_TO_TABLES = '/mnt/home/vllm_fresh/.buildkite_monitor/db_demo/'


params = {
    'created_from': LAST_24_HOURS,
    'per_page': 100,
    'include_retried_jobs': True
}

raw = fetch_data(params, BUILDKITE_API_TOKEN, ORGANIZATION_SLUG, PIPELINE_SLUG)
if raw.empty:
    print('No data to process')
    raise SystemExit

now_utc = pd.Timestamp.now(tz='UTC')
raw['timestamp'] = now_utc

now = datetime.now(zoneinfo.ZoneInfo('Europe/Helsinki')).strftime('%Y-%m-%d_%H:%M')
raw.drop(['pipeline.steps'], axis=1).to_parquet(PATH_TO_TABLES + f'raw_data/raw_data_fetch_{now}.parquet', index=False)

builds_bronze_path = PATH_TO_TABLES + 'bronze_tables/builds.parquet'
jobs_bronze_path = PATH_TO_TABLES + 'bronze_tables/jobs.parquet'
agents_bronze_path = PATH_TO_TABLES + 'bronze_tables/agents.parquet'

builds_useful_columns = ['timestamp', 'url', 'id', 'web_url', 'number', 'state', 'jobs']
jobs_useful_columns = ['build_url', 'id', 'name', 'state',  'web_url', 'soft_failed', 'created_at', 'scheduled_at', 'runnable_at', 'started_at', 'finished_at', 'expired_at', 'retried', 'retried_in_job_id', 'retries_count', 'retry_source',
       'retry_type', 'agent.id', 'agent.name', 'agent.web_url', 'agent.connection_state', 'agent.meta_data', 'log_url']

df = raw.copy()
df = types_fix(df[builds_useful_columns])

jobs = pd.json_normalize(df['jobs'].explode())
jobs = types_fix(jobs[jobs_useful_columns], jobs=True)
jobs['timestamp'] = df['timestamp'].values[0]


# Builds bronze table
df.drop(['jobs'], axis=1, inplace=True)
bronze_builds = bronze_builds(df, False)

# Jobs bronze table
bronze_jobs = bronze_jobs_logs(jobs, BUILDKITE_API_TOKEN, ORGANIZATION_SLUG, PIPELINE_SLUG, False)


# Agents bronze table
agent_columns = ['timestamp'] + [col for col in jobs.columns if col not in ['timestamp', 'log_url']]
agents = jobs[agent_columns]
agents = agents[agents.agent_name.notna()]
agents['build_number'] = agents['build_url'].apply(lambda x: x.split('/')[-1])

agents = agents[['timestamp', 'agent_name', 'agent_id', 'agent_web_url', 'agent_connection_state', 'agent_meta_data', 'build_number', 'id', 'name', 'state', 'web_url',
       'soft_failed', 'created_at', 'scheduled_at', 'runnable_at', 'started_at', 'finished_at', 'expired_at', 'retried']]

bronze_agents = bronze_agents(agents, False)    

# silver layer
jobs_silver_path = PATH_TO_TABLES + 'silver_tables/jobs.parquet'
builds_silver_path = PATH_TO_TABLES + 'silver_tables/builds.parquet'

if not os.path.exists(jobs_silver_path):
    print("Silver jobs table doesn't exist")
    silver_jobs = calculate_wait_time(bronze_jobs)
    silver_jobs.to_parquet(jobs_silver_path, index=False)
else:    
    print("Calculate waiting time only for jobs that not yet started and not cancelled, passed or failed") 