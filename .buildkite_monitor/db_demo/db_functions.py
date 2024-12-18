import requests
import pandas as pd
import os
import glob
import difflib
import time
PATH_TO_TABLES = '/mnt/home/vllm_fresh/.buildkite_monitor/db_demo/'


def fetch_data(params, token, org_slug, pipe_slug):
    # Set the URL
    url = f"https://api.buildkite.com/v2/organizations/{org_slug}/pipelines/{pipe_slug}/builds"


    # Set the headers
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    # Make the GET request
    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return pd.json_normalize(data)
    else:
        print(f"Request failed with status code {response.status_code}")
        return pd.DataFrame()


def types_fix(df, jobs=False):
    if jobs:
        df['soft_failed'] = df['soft_failed'].astype('bool')
        df['runnable_at'] = pd.to_datetime(df['runnable_at'], errors='coerce')
        df['expired_at'] = pd.to_datetime(df['expired_at'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['scheduled_at'] = pd.to_datetime(df['scheduled_at'], errors='coerce')
        df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
        df['finished_at'] = pd.to_datetime(df['finished_at'], errors='coerce')
        df.columns = df.columns.str.replace('.', '_')
    else:
        df['number'] = df['number'].astype('int')
    

    return df

def read_latest_parquet(path):
   raw_data_dir = path
   parquet_files = glob.glob(os.path.join(raw_data_dir, '*.parquet'))

   # Check if there are any parquet files in the directory
   if not parquet_files:
      raise FileNotFoundError("No Parquet files found in the directory.")

   # Sort the files by modification time in descending order
   parquet_files.sort(key=os.path.getmtime, reverse=True)

   # Get the latest parquet file
   latest_parquet_file = parquet_files[0]

   # Read the latest parquet file into a DataFrame or just use raw
   df = pd.read_parquet(latest_parquet_file)  
   return df  


def calculate_wait_time(df):
    now_utc = pd.Timestamp.now(tz='UTC')
    
    # Calculate the difference in seconds
    df['waited_seconds'] = df.apply(
        lambda row: (row['started_at'] - row['runnable_at']).total_seconds() if pd.notna(row['started_at']) and pd.notna(row['runnable_at'])  \
              else (now_utc - row['runnable_at']).total_seconds() if pd.isna(row['started_at']) and pd.notna(row['runnable_at']) and row['state']!='canceled' \
                else None,
        axis=1
    )
    
    return df

def fetch_job_log(url, token, org_slug, pipe_slug):

    # Set the headers
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    # Make the GET request
    response = requests.get(url, headers=headers)
    rate_limit_remaining = response.headers.get('RateLimit-Remaining')
    #rate_limit_limit = response.headers.get('RateLimit-Limit')
    rate_limit_reset = response.headers.get('RateLimit-Reset')
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return data['content'], rate_limit_remaining, rate_limit_reset
    else:
        print(f"Request failed with status code {response.status_code}")
        return '', rate_limit_remaining, rate_limit_reset          

def fetch_logs_for_df(df, token, org_slug, pipe_slug):
    start_time = time.time()
    for idx in df.id.unique():
         #print(j[j.id==idx].log_url.isna())
         #print(idx)
         if df[df.id==idx].log_url.notna().values[0]:
         #if j[j.id==idx].log_url.notna(): # when silver table exists, we might want to check if older jobs that didn't have logs (had nan log_url), if there is something there now.
               log, rate_limit_remaining, rate_limit_reset = fetch_job_log(df[df.id==idx].log_url.values[0], token, org_slug, pipe_slug)
               df.loc[df.id==idx, 'log'] = log
               rate_limit = int(rate_limit_remaining)
               #print(rate_limit)
               if rate_limit < 20:
                  print(f'rate_limit is low = {rate_limit}, sleeping for {rate_limit_reset} seconds')
                  time.sleep(int(rate_limit_reset))

    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    return df

def reconstruct_log(df):
    full_log = ''
    for log in df['log']:
        if log != '':
         full_log += log# + '\n'
    return full_log.strip()

def compute_diff(old_log, new_log):
    diff = difflib.unified_diff(
        old_log.splitlines(keepends=True),
        new_log.splitlines(keepends=True),
        fromfile='old_log',
        tofile='new_log'
    )
    return ''.join(diff)

builds_bronze_path = PATH_TO_TABLES + 'bronze_tables/builds.parquet'
jobs_bronze_path = PATH_TO_TABLES + 'bronze_tables/jobs.parquet'
agents_bronze_path = PATH_TO_TABLES + 'bronze_tables/agents.parquet'

def bronze_builds(df, testing):
   if not os.path.exists(builds_bronze_path):
      print("Bronze builds table doesn't exist")
      df.to_parquet(builds_bronze_path, index=False)
      return df
   else: 
      print("Bronze builds table exists, checking what is updated")
      #find updated rows, find new rows, update what needs to be updated, concat new ones and rewrite the file.
      bronze = pd.read_parquet(builds_bronze_path)  

      latest_indices = bronze.groupby(['id'])['timestamp'].idxmax()
      bronze_latest = bronze[bronze.index.isin(latest_indices)]
      #print('bronze latest timestamps only:')
      #display(bronze_latest)

      existing_builds = pd.merge(bronze_latest, df, on='id', how='inner', suffixes=('_old', ''))
      update_check = False
      if not existing_builds.empty:
         print('info about builds exists in builds bronze table')
         #latest_indices = existing_builds.groupby(['id'])['timestamp_old'].idxmax()
         #existing_builds = existing_builds[existing_builds.index.isin(latest_indices)]
         #print('filter for the latest timestamp for the builds')
         #display(existing_builds)

         comparison_columns_old = [col for col in existing_builds.columns if '_old' in col and 'timestamp' not in col]
         comparison_columns_new = [col[:-4] for col in comparison_columns_old]
         df_old = existing_builds[comparison_columns_old].sort_index(axis=1).copy()
         df_new = existing_builds[comparison_columns_new].sort_index(axis=1).copy()    
         df_old.columns = df_new.columns
         mask = df_old.fillna('None').ne(df_new.fillna('None'))
         #mask = existing_agent_jobs[comparison_columns].sort_index(axis=1).fillna('None').ne(existing_agent_jobs[comparison_columns].rename(columns=lambda x: x[:-4]).sort_index(axis=1).fillna('None'))
         #display(mask)
         updated_rows = existing_builds[mask.any(axis=1)]
         if not updated_rows.empty:
               print('There are updated rows')
               #display(updated_rows)
               columns_without_old_suffix = [col for col in updated_rows.columns if not col.endswith('_old')]
               update_check = True
               bronze = pd.concat([bronze, updated_rows[columns_without_old_suffix]], ignore_index=True)  

               #print('bronze builds with updated rows:')
               #display(bronze)

      # Identify new builds
      new_builds = df[(~df['id'].isin(bronze['id']))]
      #print('New builds')
      #display(new_builds)
      if not new_builds.empty:
      # Append new builds to the existing data
         print('There are new builds, appending them to the existing data')
         update_check = True
         #print('bronze builds with new rows:')
         bronze = pd.concat([bronze, new_builds], ignore_index=True)
         #display(bronze)


      if update_check:
         print('There were changes, write new bronze data to parquet')
         if not testing:
            bronze.to_parquet(builds_bronze_path, index=False) 
         else:
            print('final DF:')
            #display(bronze)
            
      return bronze        

def bronze_jobs_logs(df, token, org_slug, pipe_slug, testing):
   if not os.path.exists(jobs_bronze_path):
      print("Bronze jobs table doesn't exist")
      df['log'] = None
      df = fetch_logs_for_df(df, token, org_slug, pipe_slug)

      df.to_parquet(jobs_bronze_path, index=False)
      return df
   else: 
      print("Bronze jobs table exists, checking what is updated")
      #find updated rows, find new rows, update what needs to be updated, concat new ones and rewrite the file.
      bronze = pd.read_parquet(jobs_bronze_path)  

      latest_indices = bronze.groupby(['id'])['timestamp'].idxmax()
      bronze_latest = bronze[bronze.index.isin(latest_indices)]
      #print('bronze latest timestamps only:')
      #display(bronze_latest)
      df['log'] = None
      df = fetch_logs_for_df(df, token, org_slug, pipe_slug)
      #df.to_parquet(jobs_bronze_path + '_2', index=False)
      # maybe possible to check if states are passed in bronze and df for the same job, then we don't need to fetch the log
      existing_jobs = pd.merge(bronze_latest, df, on='id', how='inner', suffixes=('_old', ''))
      update_check = False
      if not existing_jobs.empty:
         print('info about jobs exists in jobs bronze table')
         #latest_indices = existing_builds.groupby(['id'])['timestamp_old'].idxmax()
         #existing_builds = existing_builds[existing_builds.index.isin(latest_indices)]
         #print('filter for the latest timestamp for the builds')
         #display(existing_builds)

         comparison_columns_old = [col for col in existing_jobs.columns if '_old' in col and 'timestamp' not in col and 'agent_meta_data' not in col]
         comparison_columns_new = [col[:-4] for col in comparison_columns_old]
         df_old = existing_jobs[comparison_columns_old].copy()#.sort_index(axis=1).copy()
         df_old_columns = [col[:-4] for col in df_old.columns]
         df_old.columns = df_old_columns
         df_old.sort_index(axis=1, inplace=True)
         df_new = existing_jobs[comparison_columns_new].sort_index(axis=1).copy()    
         assert (df_old.columns == df_new.columns).all()
         mask = df_old.fillna('None').ne(df_new.fillna('None'))
         #mask = existing_agent_jobs[comparison_columns].sort_index(axis=1).fillna('None').ne(existing_agent_jobs[comparison_columns].rename(columns=lambda x: x[:-4]).sort_index(axis=1).fillna('None'))
         #display(mask.loc[mask.any(axis=1), mask.any(axis=0)])
         #print('mask cols', mask.columns)
         updated_rows = existing_jobs[mask.any(axis=1)]
         if not updated_rows.empty:
               print('There are updated rows')
               #display(updated_rows)
               columns_without_old_suffix = [col for col in updated_rows.columns if not col.endswith('_old')]
               update_check = True

               #print('The following jobs have changes in log:')

               #display(existing_jobs[mask.log==True])
               #print('jobs with updated log:', existing_jobs[mask.log==True].id.unique())
               for idx in existing_jobs[mask.log==True].id:
                  #print('id', idx)
                  reconstructed_log = reconstruct_log(bronze[bronze.id==idx].sort_values('timestamp'))
                  #print(reconstructed_log)
                  new_log = updated_rows[updated_rows.id==idx].log.values[0]
                  #print(new_log)
                  diff = compute_diff(reconstructed_log, new_log)
                  #print('diff', diff)
                  added_lines = [line[1:] for line in diff.splitlines() if line.startswith('+') and not line.startswith('+++')]
                  if added_lines:
                     #print('added lines:', added_lines)
                  #df2.loc[0, 'log'] = '\n'.join(added_lines)
                     #print('updated_rows with the id:', updated_rows.loc[updated_rows.id==idx])
                     updated_rows.loc[updated_rows.id==idx, 'log'] = '\n'.join(added_lines)
                     #print('updated_rows', updated_rows)
                  
               bronze = pd.concat([bronze, updated_rows[columns_without_old_suffix]], ignore_index=True)  
               # I need to check if log was changed, then I need to compare it to existing log for the job and store only diff
               

               #print('bronze jobs tail with updated rows:')
               #display(bronze.tail())

      # Identify new jobs
      new_jobs = df[(~df['id'].isin(bronze['id']))]
      print('New jobs')
      #display(new_jobs)
      if not new_jobs.empty:
      # Append new jobs to the existing data
         print('There are new jobs, appending them to the existing data')
         update_check = True
         #print('bronze jobs with new rows:')
         bronze = pd.concat([bronze, new_jobs], ignore_index=True)
         #display(bronze.head())


      if update_check:
         print('There were changes, write new bronze data to parquet')
         if not testing:
            bronze.to_parquet(jobs_bronze_path, index=False) 
         #else:
            #print('final DF:')
            #display(bronze.head())
            
      return bronze        


def bronze_agents(df, testing):
   if not os.path.exists(agents_bronze_path):
      print("Bronze agents table doesn't exist")
      df.to_parquet(agents_bronze_path, index=False)
      return df
   else: 
      print("Bronze agents table exists, checking what is updated")
      #find updated rows, find new rows, update what needs to be updated, concat new ones and rewrite the file.
      bronze = pd.read_parquet(agents_bronze_path)  

      latest_indices = bronze.groupby(['agent_name', 'id'])['timestamp'].idxmax()
      bronze_latest = bronze[bronze.index.isin(latest_indices)]
      #print('bronze latest timestamps only:')
      #display(bronze_latest)

      existing_agents = pd.merge(bronze_latest, df, on=['agent_name', 'id'], how='inner', suffixes=('_old', ''))
      update_check = False
      if not existing_agents.empty:
         print('info about agents exists in agents bronze table')
         #latest_indices = existing_builds.groupby(['id'])['timestamp_old'].idxmax()
         #existing_builds = existing_builds[existing_builds.index.isin(latest_indices)]
         #print('filter for the latest timestamp for the builds')
         #display(existing_builds)

         comparison_columns_old = [col for col in existing_agents.columns if '_old' in col and 'timestamp' not in col and 'agent_meta_data' not in col] 
         comparison_columns_new = [col[:-4] for col in comparison_columns_old]
         df_old = existing_agents[comparison_columns_old].sort_index(axis=1).copy()
         df_new = existing_agents[comparison_columns_new].sort_index(axis=1).copy()    
         df_old.columns = df_new.columns
         mask = df_old.fillna('None').ne(df_new.fillna('None'))
         #mask = existing_agent_jobs[comparison_columns].sort_index(axis=1).fillna('None').ne(existing_agent_jobs[comparison_columns].rename(columns=lambda x: x[:-4]).sort_index(axis=1).fillna('None'))
        # display(mask)
         updated_rows = existing_agents[mask.any(axis=1)]
         if not updated_rows.empty:
               print('There are updated rows')
               #display(updated_rows)
               columns_without_old_suffix = [col for col in updated_rows.columns if not col.endswith('_old')]
               update_check = True
               bronze = pd.concat([bronze, updated_rows[columns_without_old_suffix]], ignore_index=True)  

               #print('bronze jobs with updated rows:')
               #display(bronze)

      # Identify new agents
      new_agents = df[(~df['id'].isin(bronze['id']))]
      #print('New agents')
      #display(new_agents)
      if not new_agents.empty:
      # Append new agents to the existing data
         print('There are new agents, appending them to the existing data')
         update_check = True
         #print('bronze agents with new rows:')
         bronze = pd.concat([bronze, new_agents], ignore_index=True)
         #display(bronze)


      if update_check:
         print('There were changes, write new bronze data to parquet')
         if not testing:
            bronze.to_parquet(agents_bronze_path, index=False) 
         #else:
            #print('final DF:')
            #display(bronze)
            
      return bronze              