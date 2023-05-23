import os
import mimetypes
import time
import logging
import openai
import dotenv
from flask import Flask, request, jsonify
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
