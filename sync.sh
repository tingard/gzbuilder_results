gcloud compute scp --project gzbuilder --zone europe-west4-b --recurse fit_models.py gzb-model-fitting-vm:~/ && gcloud compute scp --project gzbuilder --zone europe-west4-b --recurse affirmation_subjects_results/do_fit.py gzb-model-fitting-vm:~/affirmation_subjects_results/
