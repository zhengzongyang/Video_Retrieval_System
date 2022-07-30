from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules



class VrAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'VR_APP'
    
    def ready(self) -> None:
        autodiscover_modules('preload.py')
        return
