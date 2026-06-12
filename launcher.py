"""Launcher - double click this file to run the demo"""
import subprocess, sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(script_dir, '完整演示.py')

subprocess.run([sys.executable, script_path], cwd=script_dir)
input('\nPress Enter to exit...')
