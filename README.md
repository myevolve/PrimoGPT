# PrimoGPT

Važno je spomenuti da sam skripte za PrimoGPT naspisao sam, da se radi o RAG tehnici gdje sam osmislio pristup s generiranjem značajki. Isnpiraciju sam dobio iz:

Kod firnrl-a je važno npomenuti da je to fork ali da sam ja modificirao ove primo env.

Vezano za primorl:
env_primo_default je isti kao i finrl env_stocktrading.py samo su dodani custom logovi kako bi se lakse mogli pratiti rezultati

env_primo_default_nlp je isti kao i env_primo_default samo da je dodana NLP funkcionalnost.

env_primorl je novi env koji je nastao na temelju env_stocktrading.py s učitavanjem dodatnih NLP značajki u custom definiranom funkxijom nagrade.

Kod PrimoGPT-a sam napisao prepare_data.py i create_prompt.py. TO je taj svojevrsni rag. Hendlao sam pripremu podataka za trening, korištenje custom modela. 


Moram dodati reference koje sam koristio.

Zahvalti se fundaciji.
