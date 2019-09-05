#!/bin/bash
#SBATCH --job-name=rsscnn       # Nombre del trabajo
#SBATCH --output=output/vgg_safety_%j.log         # Nombre del output (%j se reemplaza por el ID del trabajo)
#SBATCH --error=output/err/vgg_safety_%j.err          # Output de errores (opcional)
#SBATCH --ntasks=1                   # Correr 2 tareas
#SBATCH --cpus-per-task=4            # Numero de cores por tarea
#SBATCH --distribution=cyclic:cyclic # Distribuir las tareas de modo ciclico
#SBATCH --time=4-00:00:00            # Timpo limite d-hrs:min:sec
#SBATCH --mem-per-cpu=6000mb         # Memoria por proceso
#SBATCH --mail-type=END,FAIL         # Enviar eventos al mail (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=afcadiz@uc.cl    # El mail del usuario
#SBATCH --partition=ialab-high        # Se tiene que elegir una partición de nodos con GPU
#SBATCH --gres=gpu:1                 # Usar 2 GPUs (se pueden usar N GPUs de marca especifica de la manera --gres=gpu:marca:N)
#SBATCH --nodelist=hydra


pyenv/bin/python3 train.py  --model rscnn --max_epochs 20 --premodel vgg --attribute safety --wd 0.0001 --lr 0.001  --batch_size 8 --dataset /vault/ironcadiz/placepulse --csv /vault/ironcadiz/votes_clean.csv


