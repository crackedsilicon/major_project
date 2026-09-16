from pathlib import Path
import shutil
import sys

sys.path.insert(0, '.')

from scripts.prepare_data import prepare


SOURCE = Path('data/raw/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_attack_Bro/normal_DDoS/normal_DDoS_11')
TARGET = Path('data/raw/TON_IoT_small')


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for csv_file in SOURCE.glob('*.csv'):
        shutil.copy2(csv_file, TARGET / csv_file.name)
        print(f'Copied {csv_file.name}')

    prepare('TON_IoT_small', 'data/processed', n_clients=3, split='iid', seed=42)


if __name__ == '__main__':
    main()
