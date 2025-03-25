#views.py
from django.shortcuts import render
from django.http import HttpResponse
from . import malware_pred
from django.shortcuts import render
from webpage.tasks import Command
# Example: Inside a function in views.py
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType

def index(request):
    return render(request, 'index.html')

def run_ml_models(request):
    if request.method == 'POST' and request.FILES['file']:
        Command().handle()
        print("Command executed")
        file = request.FILES['file']
        file_content = file.read().decode('utf-8')
        print(file_content)
        prediction = malware_pred.select_file_and_predict(file_content)
        print(prediction)
        return render(request, 'result.html', {'prediction': prediction})
    else:
        return HttpResponse('No file selected. Exiting...')
