import re
from collections import defaultdict
import matplotlib.pyplot as plt


# Fonction pour analyser le fichier
def parse_losses(filename):
    epoch_data = defaultdict(lambda: defaultdict(list))
    epoch_index = 0

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Ignorer les lignes vides ou parasites
            if not line or re.match(r'^\d+it \[.*?\]$', line):
                continue

            # Identifier les pertes et les enregistrer
            cumulative_match = re.match(r'cumulative (\d+\.?\d*)', line)
            grounding_match = re.match(r'grounding tensor\((\d+\.?\d*)', line)
            denoise_match = re.match(r'denoise tensor\((\d+\.?\d*)', line)

            if cumulative_match:
                epoch_data[epoch_index]["cumulative"].append(float(cumulative_match.group(1)))
            elif grounding_match:
                epoch_data[epoch_index]["grounding"].append(float(grounding_match.group(1)))
            elif denoise_match:
                epoch_data[epoch_index]["denoise"].append(float(denoise_match.group(1)))

            # Détecter la fin d'une époque
            if line.startswith("epochloss:"):
                epoch_index += 1

    # Calculer les moyennes par époque
    epoch_averages = {}
    for epoch, losses in epoch_data.items():
        epoch_averages[epoch] = {key: sum(values) / len(values) for key, values in losses.items() if values}

    return epoch_averages


filename = "output.txt"
epoch_averages = parse_losses(filename)

# Afficher les résultats
for epoch, averages in epoch_averages.items():
    print(f"Epoch {epoch}:")
    for loss_type, avg in averages.items():
        print(f"  {loss_type.capitalize()} Loss: {avg:.4f}")

last_epoch = max(epoch_averages.keys())  # Trouver la dernière époque
epoch_averages.pop(last_epoch, None)


def normalize_losses(losses):
    min_loss = min(losses)
    max_loss = max(losses)
    normalized = [(loss - min_loss) / (max_loss - min_loss) for loss in losses]
    return normalized


# Remplacez ce bloc pour afficher les résultats sous forme de courbes
def plot_losses(epoch_averages):
    epochs = sorted(epoch_averages.keys())  # Trier les époques
    loss_types = epoch_averages[epochs[0]].keys()  # Types de pertes disponibles

    # Tracer chaque type de perte
    for loss_type in loss_types:
        values = [epoch_averages[epoch][loss_type] for epoch in epochs]
        plt.plot(epochs, normalize_losses(values), label=loss_type.capitalize())

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Normalized Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.show()

# Utilisation
plot_losses(epoch_averages)
