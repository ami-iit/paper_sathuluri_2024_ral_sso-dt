# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Computer Modern Roman"
plt.rcParams["axes.grid"] = True
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["figure.dpi"] = 300


seed = 42
rng = np.random.default_rng(seed=seed)
T_sim = 20

filetag = "omega/18012025032840"
subfiletag = "053023"
m = "offset"

# %%
collected_sample_files = glob.glob(
    f"../optimisation_logs/{filetag}_collected_samples_{m}/{subfiletag}/*.pkl"
)
print(len(collected_sample_files))

# %%
all_samples_temp = []
for file in collected_sample_files:
    with open(file, "rb") as f:
        expt_data = pickle.load(f)
        all_samples_temp.append(expt_data)
all_samples = np.vstack(all_samples_temp)
p = rng.permutation(len(all_samples))
all_samples = all_samples[p]
out_dats = all_samples[:, [-1, -4, -3, -2]]
xs = all_samples[:, :-4]


def plot_data_dist_hist(plt_data):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].hist(plt_data[:, 0])
    axs[0, 0].set_title("Histogram of fitness")
    axs[0, 1].hist(plt_data[:, 1])
    axs[0, 1].set_title("Histogram of sim time")
    axs[1, 0].hist(plt_data[:, 2])
    axs[1, 0].set_title("Histogram of $P_f$")
    axs[1, 1].hist(plt_data[:, 3])
    axs[1, 1].set_title("Histogram of $P_j$")
    plt.tight_layout()
    plt.show()


plot_data_dist_hist(plt_data=out_dats)

# %%
selected_samples_x = xs
selected_samples_out_dat = out_dats
plot_data_dist_hist(plt_data=selected_samples_out_dat)
print("Loading all collected samples")

good_size = len(selected_samples_x[selected_samples_out_dat[:, 1] >= T_sim])
total_size = len(selected_samples_x)

# %%

Xs = selected_samples_x
Ys = (selected_samples_out_dat[:, 1] == T_sim).astype(int).reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
scaled_X = scaler_X.fit_transform(Xs)
scaled_Y = scaler_Y.fit_transform(Ys)

X_train, X_test, Y_train, Y_test = train_test_split(
    scaled_X, scaled_Y, test_size=0.3, random_state=seed, shuffle=True
)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# %%
config = {
    "num_layers": 1,
    "num_neurons": 64,
    "solver": "adam",
    "alpha": 1e-3,
    "activation": "relu",
}
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(config["num_neurons"],) * config["num_layers"],
    activation=config["activation"],
    solver=config["solver"],
    alpha=config["alpha"],
    learning_rate="adaptive",
    max_iter=200,
    random_state=seed,
)
mlp_clf.fit(X_train, Y_train)

# %%
print("Accuracy:", mlp_clf.score(X_test, Y_test))
cm = confusion_matrix(Y_test, mlp_clf.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp_clf.classes_)
disp.plot()
plt.savefig(
    f"../assets/confusion_matrix.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
with open(f"../assets/classifier_{m}_{good_size}_{total_size}.pkl", "wb") as f:
    pickle.dump(mlp_clf, f)
with open(f"../assets/scaler_{m}_{good_size}_{total_size}.pkl", "wb") as f:
    pickle.dump(scaler_X, f)
