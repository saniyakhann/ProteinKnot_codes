import requests

#test proteins 
test_proteins = ["1crn", "4hhb", "5zeu"]

for prot in test_proteins:
    url = f'https://files.rcsb.org/download/{prot}.pdb'
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"{prot}: Found")
        else:
            print(f"{prot}: Not found (Status {response.status_code})")
    except Exception as e:
        print(f"{prot}: Error - {e}")
