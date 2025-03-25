# tasks.py

from django.core.management.base import BaseCommand
from webpage.email_alerts import process_intruder_alert

class Command(BaseCommand):
    help = 'Runs face recognition and sends email alert for intruder detection'

    def handle(self, *args, **options):
        process_intruder_alert()
