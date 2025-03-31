from tensorflow import keras


def build_cnn(
    n_setups: int,
    contains_embedding_layer: bool,
    maxlen: int,
    n_aa: int,
    n_cnn_layers: int,
    n_filters: int,
    kernel_size: int,
    cnn_activation: str,
    n_dense: int,
    dense_layer_size: int,
    dropout_rate: float,
    learning_rate: float,
    embedding_vector_length: int,
    batch_normalisation: bool,
    loss: str,
):
    """
    Build a CNN model for multiple setups.
    Args:
        n_setups (int): Number of setups: this is the number of "parallel" models.
        contains_embedding_layer (bool): Whether to include an embedding layer.
        maxlen (int): Maximum length of the input sequences.
        n_aa (int): Length of the encoded amino acid, this would be a label for embedding layer and for instance 22 for one-hot encoding.
        n_dense (int): Number of dense layers.
        n_cnn_layers (int): Number of CNN layers.
        n_filters (int): Number of filters in the CNN layers.
        kernel_size (int): Kernel size for the CNN layers.
        cnn_activation (str): Activation function for the CNN layers.
        dropout_rate (float): Dropout rate for the layers.
        learning_rate (float): Learning rate for the optimizer.
        embedding_vector_length (int): Length of the embedding vector.
        batch_normalisation (bool): Whether to include batch normalization.
        loss (str): Loss function for the model.
    Returns:
        model (keras.Model): Compiled Keras model.
    """

    inputs = [keras.layers.Input(shape=(maxlen, n_aa)) for _ in range(n_setups)]

    xs = inputs
    if contains_embedding_layer:
        inputs = [keras.layers.Input(shape=(maxlen,)) for _ in range(n_setups)]
        xs = [
            keras.layers.Embedding(
                n_aa, embedding_vector_length, input_length=maxlen, mask_zero=True
            )(inp)
            for inp in inputs
        ]

    def add_cnns(x):
        for layer_ix in range(n_cnn_layers):
            x = keras.layers.Conv1D(
                filters=(layer_ix + 1) * n_filters,
                kernel_size=kernel_size,
                strides=1,
                activation=cnn_activation,
                padding="same",
            )(x)
            if batch_normalisation:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(dropout_rate)(x)
        return x

    xs = [add_cnns(x) for x in xs]

    xs = [keras.layers.MaxPooling1D()(x) for x in xs]

    xs = [keras.layers.Flatten()(x) for x in xs]

    def add_denses(x, n_dense, dense_layer_size, dropout_rate):
        for layer_ix in range(n_dense):
            x = keras.layers.Dense(
                max(dense_layer_size // (2**layer_ix), 1),
                activation="relu",
            )(x)
            if batch_normalisation:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(dropout_rate)(x)
        return x

    xs = [add_denses(x, n_dense, dense_layer_size, dropout_rate) for x in xs]
    outputs = [
        keras.layers.Dense(1, activation="sigmoid", name=f"setup_{ix}")(x)
        for ix, x in enumerate(xs)
    ]
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss={f"setup_{ix}": loss for ix in range(n_setups)},
        loss_weights=[1 / n_setups] * n_setups,
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        optimizer=opt,
    )

    return model


def build_gru(
    n_setups: int,
    contains_embedding_layer: bool,
    maxlen: int,
    n_aa: int,
    n_gru: int,
    gru_layer_size: int,
    learning_rate: float,
    dropout_rate: float,
    embedding_vector_length: int,
    batch_normalisation: bool,
    loss: str,
):
    """
    Build a GRU model for multiple setups.
    Args:
        n_setups (int): Number of setups: this is the number of "parallel" models.
        contains_embedding_layer (bool): Whether to include an embedding layer.
        maxlen (int): Maximum length of the input sequences.
        n_aa (int): Length of the encoded amino acid, this would be a label for embedding layer and for instance 22 for one-hot encoding.
        n_dense (int): Number of dense layers.
        n_gru (int): Number of GRU layers.
        gru_layer_size (int): Number of units in the GRU layers.
        dropout_rate (float): Dropout rate for the layers.
        learning_rate (float): Learning rate for the optimizer.
        embedding_vector_length (int): Length of the embedding vector.
        batch_normalisation (bool): Whether to include batch normalization.
        loss (str): Loss function for the model.
    Returns:
        model (keras.Model): Compiled Keras model.
    """
    inputs = [keras.layers.Input(shape=(maxlen, n_aa)) for _ in range(n_setups)]

    xs = inputs
    if contains_embedding_layer:
        inputs = [keras.layers.Input(shape=(maxlen,)) for _ in range(n_setups)]
        xs = [
            keras.layers.Embedding(
                n_aa, embedding_vector_length, input_length=maxlen, mask_zero=True
            )(inp)
            for inp in inputs
        ]

    def add_grus(x, n_gru, gru_layer_size, dropout_rate):
        for layer_ix in range(n_gru):
            x = keras.layers.GRU(gru_layer_size, return_sequences=True)(x)
            if batch_normalisation:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(dropout_rate)(x)
        return x

    xs = [add_grus(x, n_gru, gru_layer_size, dropout_rate) for x in xs]
    xs = [keras.layers.Flatten()(x) for x in xs]

    outputs = [
        keras.layers.Dense(1, activation="sigmoid", name=f"setup_{ix}")(x)
        for ix, x in enumerate(xs)
    ]
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss={f"setup_{ix}": loss for ix in range(n_setups)},
        loss_weights=[1 / n_setups] * n_setups,
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        optimizer=opt,
    )
    return model


def build_mlp(
    n_setups: int,
    contains_embedding_layer: bool,
    maxlen: int,
    n_aa: int,
    n_dense: int,
    dense_layer_size: int,
    learning_rate: float,
    dropout_rate: float,
    embedding_vector_length: int,
    batch_normalisation: bool,
    loss: str,
):
    """
    Build a simple MLP model for multiple setups.
    Args:
        n_setups (int): Number of setups: this is the number of "parallel" models.
        contains_embedding_layer (bool): Whether to include an embedding layer.
        maxlen (int): Maximum length of the input sequences.
        n_aa (int): Length of the encoded amino acid, this would be a label for embedding layer and for instance 22 for one-hot encoding.
        n_dense (int): Number of dense layers.
        dense_layer_size (int): Number of units in the dense layers.
        dropout_rate (float): Dropout rate for the layers.
        learning_rate (float): Learning rate for the optimizer.
        embedding_vector_length (int): Length of the embedding vector.
        batch_normalisation (bool): Whether to include batch normalization.
        loss (str): Loss function for the model.
    Returns:
        model (keras.Model): Compiled Keras model.
    """

    inputs = [keras.layers.Input(shape=(maxlen, n_aa)) for _ in range(n_setups)]

    xs = inputs
    if contains_embedding_layer and embedding_vector_length != 0:
        inputs = [
            keras.layers.Input(shape=(maxlen,)) for _ in range(n_setups)
        ]  # dan krijg ik genoeg input layers
        xs = [
            keras.layers.Embedding(
                n_aa, embedding_vector_length, input_length=maxlen, mask_zero=False
            )(inp)
            for inp in inputs
        ]

    xs = [keras.layers.Flatten()(x) for x in xs]

    def add_denses(x, n_dense, dense_layer_size, dropout_rate):
        for layer_ix in range(n_dense):
            x = keras.layers.Dense(
                max(dense_layer_size // (2**layer_ix), 1),
                activation="relu",
            )(x)
            if batch_normalisation:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(dropout_rate)(x)
        return x

    xs = [add_denses(x, n_dense, dense_layer_size, dropout_rate) for x in xs]
    outputs = [
        keras.layers.Dense(1, activation="sigmoid", name=f"setup_{ix}")(x)
        for ix, x in enumerate(xs)
    ]
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss={f"setup_{ix}": loss for ix in range(n_setups)},
        loss_weights=[1 / n_setups] * n_setups,
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        optimizer=opt,
    )
    return model


def build_model(
    model_name: str,
    row,
    dict_encoding,
    batch_normalisation,
    max_sequence_length,
    alphabet,
    n_setups,
    row_n,
    embedding_layer,
    len_encoding,
):
    """
    Build a model based on the specified model name and parameters.
    Args:
        model_name (str): Name of the model to build ('mlp', 'cnn', 'gru').
        row (pd.Series): Row containing the model parameters.
        dict_encoding (dict): Dictionary for encoding amino acids.
        batch_normalisation (bool): Whether to use batch normalization.
        max_sequence_length (int): Maximum sequence length for padding.
        alphabet (list): List of amino acids.
        row_n (int): Row number for configuration.
        n_setups (int): Number of setups.
        embedding_layer (bool): Whether to include an embedding layer.
        len_encoding (int): Length of the encoded amino acid.
    Returns:
        model (keras.Model): Compiled Keras model.
    """
    if model_name == "mlp":
        dictionary = {
            "configuration_number": row_n,
            "contains_embedding_layer": embedding_layer,
            "model_type": row["type_of_model"],
            "n_dense_layers": int(row["n_dense_layers"]),
            "learning_rate": row["learning_rate"],
            "dropout": row["dropout"],
            "fit_batch_size": int(row["fit_batch_size"]),
            "loss": "binary_crossentropy",
            "optimizer": row["optimizer"],
            "fit_epochs": int(row["fit_epochs"]),
            "n_dense_neurons": int(row["n_dense_neurons"]),
            "embedding_vector_length": int(row["embedding_vector_length"]),
            "batch_normalisation": str(batch_normalisation),
            "dict_encoding": str(dict_encoding),
        }

        model = build_mlp(
            n_setups=n_setups,
            contains_embedding_layer=embedding_layer,
            maxlen=max_sequence_length,
            n_aa=len_encoding,
            n_dense=dictionary["n_dense_layers"],
            dense_layer_size=dictionary["n_dense_neurons"],
            learning_rate=float(row["learning_rate"]),
            dropout_rate=float(row["dropout"]),
            embedding_vector_length=int(row["embedding_vector_length"]),
            batch_normalisation=batch_normalisation,
            loss="binary_crossentropy",
        )

    elif model_name == "cnn":
        dictionary = {
            "configuration_number": row_n,
            "contains_embedding_layer": True,
            "maxlen": max_sequence_length,
            "n_cnn_layers": int(row["n_cnn_layers"]),
            "n_filters": int(row["n_filters"]),
            "kernel_size": int(row["kernel_size"]),
            "cnn_activation": "relu",
            "n_dense": int(row["n_dense"]),
            "dense_layer_size": int(row["dense_layer_size"]),
            "dense_activation": "relu",
            "dropout_rate": float(row["dropout_rate"]),
            "learning_rate": float(row["learning_rate"]),
            "embedding_vector_length": int(row["embedding_vector_length"]),
            "loss": "binary_crossentropy",
            "optimizer": row["optimizer"],
            "fit_epochs": int(row["fit_epochs"]),
            "dict_encoding": str(dict_encoding),
            "batch_normalisation": str(batch_normalisation),
        }

        model = build_cnn(
            n_setups=n_setups,
            contains_embedding_layer=embedding_layer,
            maxlen=int(max_sequence_length),
            n_aa=len_encoding,
            n_cnn_layers=int(dictionary["n_cnn_layers"]),
            n_filters=dictionary["n_filters"],
            kernel_size=dictionary["kernel_size"],
            cnn_activation=dictionary["cnn_activation"],
            n_dense=dictionary["n_dense"],
            dense_layer_size=int(dictionary["dense_layer_size"]),
            dropout_rate=float(dictionary["dropout_rate"]),
            learning_rate=float(dictionary["learning_rate"]),
            embedding_vector_length=int(dictionary["embedding_vector_length"]),
            batch_normalisation=batch_normalisation,
            loss="binary_crossentropy",
        )

    elif model == "gru":
        dictionary = {
            "configuration_number": row_n,
            "contains_embedding_layer": embedding_layer,
            "maxlen": max_sequence_length,
            "n_gru": int(row["n_dense"]),
            "gru_layer_size": int(row["dense_layer_size"]),
            "dropout_rate": float(row["dropout_rate"]),
            "learning_rate": float(row["learning_rate"]),
            "embedding_vector_length": int(row["embedding_vector_length"]),
            "batch_normalisation": str(batch_normalisation),
            "loss": "binary_crossentropy",
            "optimizer": row["optimizer"],
            "fit_epochs": int(row["fit_epochs"]),
            "dict_encoding": str(dict_encoding),
        }
        model = build_gru(
            n_setups=n_setups,
            contains_embedding_layer=embedding_layer,
            maxlen=int(max_sequence_length),
            n_aa=len_encoding,
            n_gru=int(dictionary["n_gru"]),
            gru_layer_size=int(dictionary["gru_layer_size"]),
            dropout_rate=float(dictionary["dropout_rate"]),
            learning_rate=float(dictionary["learning_rate"]),
            embedding_vector_length=int(dictionary["embedding_vector_length"]),
            batch_normalisation=batch_normalisation,
            loss="binary_crossentropy",
        )

    return dictionary, model
