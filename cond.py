import yaml

def extract_packages(env_file):
    with open(env_file, 'r') as file:
        env = yaml.safe_load(file)
    packages = env.get('dependencies', [])
    # Extract only the package names (without versions)
    package_names = set()
    for package in packages:
        if isinstance(package, str):
            package_names.add(package.split('=')[0])
        elif isinstance(package, dict) and 'pip' in package:
            for pip_package in package['pip']:
                package_names.add(pip_package.split('==')[0])
    return package_names

env1_packages = extract_packages('/home/naveen/module5/regression-py/deploy/conda_envs/ct-hpp-dev.yml')
env2_packages = extract_packages('/home/naveen/module5/regression-py/deploy/conda_envs/addon-tareg-dev.yml')

# Find packages in env1 but not in env2
unique_packages = env1_packages - env2_packages

print("Packages in env1 but not in env2:")
for package in unique_packages:
    print(package)