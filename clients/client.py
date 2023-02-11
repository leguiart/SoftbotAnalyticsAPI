
import requests
import logging
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(f"__main__.{__name__}")

class GenericHttpClient:
    
    def create_session(self):
        session = requests.Session()
        retry = Retry(backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def generic_get(self, base_url, *args, **kwargs):
        route : str = os.path.join(base_url, *args)
        if len(kwargs) > 0:
            route += '?'
            for k, v in kwargs.items():
                if type(v) is list:
                    route += '&'.join([f'{k}={val}' for val in v])
                elif type(v) is dict:
                    # Common prefix key-vals
                    route += '&'.join([f'{k}_{k1}={v1}' for k1, v1 in v.items()])
                else:
                    route += f'{k}={v}'
                route += '&'
            route = route[:len(route) - 1]
        session = self.create_session()
        logger.info(f'Start GET request {route}')
        r = session.get(route)
        json_r = r.json()
        logger.info(f'Finish GET request {route}')
        return json_r
        
