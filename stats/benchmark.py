from dataclasses import dataclass
from math import ceil
import multiprocessing

import pickle
import numpy as np
from tqdm import tqdm
from usearch.eval import self_recall, measure_seconds
from usearch.index import (
    Index,
    MetricKind,
    ScalarKind,
    CompiledMetric,
    MetricSignature,
)

from usearchmolecules.dataset import FingerprintedEntry, FingerprintedDataset

from usearchmolecules.to_fingerprint import (
    FingerprintShape,
    smiles_to_maccs_ecfp4_fcfp4,
    smiles_to_pubchem,
)

from usearchmolecules.metrics_numba import (
    tanimoto_maccs,
    tanimoto_ecfp4,
    tanimoto_mixed,
    tanimoto_conditional,
)


@dataclass
class EvalResult:
    add_speed: np.ndarray
    search_speed: np.ndarray

    recall_vector: np.ndarray
    recall_molecule: np.ndarray

    memory_usage: np.ndarray
    number_of_levels: np.ndarray
    number_of_edges: np.ndarray


def eval(
    metric,
    data: tuple,
    shape: FingerprintShape,
    batch_size: int = 10000,
    recall_at: int = 1,
    num_threads: int = 100,
    end_to_end_experiments: int = 100,
):
    data_size = len(data[0])
    batch_count = int(ceil(data_size / batch_size))

    result = EvalResult(
        add_speed=np.zeros(batch_count),
        search_speed=np.zeros(batch_count),
        recall_vector=np.zeros(batch_count),
        recall_molecule=np.zeros(batch_count),
        memory_usage=np.zeros(batch_count),
        number_of_levels=np.zeros(batch_count),
        number_of_edges=np.zeros(batch_count),
    )

    index = Index(ndim=shape.nbits, metric=metric, dtype=ScalarKind.B1)

    pbar = tqdm(range(data_size))
    for batch_idx in range(batch_count):
        slice_start = batch_size * batch_idx
        slice_end = min(slice_start + batch_size, data_size)

        slice_labels = data[1][slice_start:slice_end]
        slice_vectors = data[2][slice_start:slice_end]

        add_time, _ = measure_seconds(
            lambda: index.add(slice_labels, slice_vectors, threads=num_threads)
        )
        result.add_speed[batch_idx] = batch_size / add_time

        # Sample some tasks from present elements
        search_indexes = np.random.randint(0, slice_end, size=batch_size)
        search_labels = data[1][search_indexes]
        search_vectors = data[2][search_indexes]

        search_time, search_result = measure_seconds(
            lambda: index.search(search_vectors, count=recall_at, threads=num_threads)
        )
        result.search_speed[batch_idx] = batch_size / search_time
        result.recall_vector[batch_idx] = search_result.mean_recall(search_labels)

        stats = index.levels_stats
        result.memory_usage[batch_idx] = index.memory_usage
        result.number_of_edges[batch_idx] = sum(s.edges for s in stats)
        result.number_of_levels[batch_idx] = index.nlevels

        fingers = []
        search_smiles = search_indexes[:end_to_end_experiments]
        for i in search_smiles:
            smi = str(data[0][i])
            maccs, ecfp4, fcfp4 = smiles_to_maccs_ecfp4_fcfp4(smi)
            entry = FingerprintedEntry.from_parts(smi, maccs, ecfp4, fcfp4, shape)
            fingers.append(entry.fingerprint)
        fingers = np.vstack(fingers)
        fingers_result = index.search(fingers, count=recall_at, threads=num_threads)
        result.recall_molecule[batch_idx] = fingers_result.mean_recall(search_smiles)

        pbar.update(batch_size)

    pbar.close()
    print(repr(index))
    return result


def eval_combinations(names, datas, metric, shape, batch_size):
    for name, data in zip(names, datas):
        if data is None:
            continue
        print("Starting benchmark for:", name)
        eval_result = eval(
            metric=metric,
            data=data,
            shape=shape,
            batch_size=batch_size,
            num_threads=num_threads,
        )
        with open(name + ".pickle", "wb") as f:
            pickle.dump(eval_result, f)


names_maccs = ["MACCS: PubChem", "MACCS: GDB13", "MACCS: REAL"]
names_ecfp4 = ["ECFP4: PubChem", "ECFP4: GDB13", "ECFP4: REAL"]
names_mixed = ["MACCS+ECFP4: PubChem", "MACCS+ECFP4: GDB13", "MACCS+ECFP4: REAL"]
names_conditional = ["MACCS*ECFP4: PubChem", "MACCS*ECFP4: GDB13", "MACCS*ECFP4: REAL"]

batch_size = 100_000
max_molecules = 10_000_000
max_shards = 15
num_threads = multiprocessing.cpu_count()

if __name__ == "__main__":
    chunks_pubchem = FingerprintedDataset.open("data/pubchem", max_shards=max_shards)
    chunks_gdb13 = FingerprintedDataset.open("data/gdb13", max_shards=max_shards)
    chunks_real = FingerprintedDataset.open("data/real", max_shards=max_shards)

    shape_maccs = FingerprintShape(include_maccs=True, nbytes_padding=3)
    shape_ecfp4 = FingerprintShape(include_ecfp4=True)
    shape_mixed = FingerprintShape(
        include_maccs=True, include_ecfp4=True, nbytes_padding=3
    )

    path_numba = "stats/numba/"
    path_simsimd = "stats/simsimd/"

    distance_simsimd = MetricKind.Jaccard
    distances_numba = [
        tanimoto_maccs.address,
        tanimoto_ecfp4.address,
        tanimoto_mixed.address,
        tanimoto_conditional.address,
    ]

    for use_numba in [False, True]:
        names_prefixes = path_numba if use_numba else path_simsimd

        # MACCS fingerprints
        data_pubchem_maccs = chunks_pubchem.head(
            max_molecules, shape=shape_maccs, shuffle=True
        )
        data_gdb13_maccs = chunks_gdb13.head(
            max_molecules, shape=shape_maccs, shuffle=True
        )
        data_real_maccs = chunks_real.head(
            max_molecules, shape=shape_maccs, shuffle=True
        )

        datas_maccs = [data_pubchem_maccs, data_gdb13_maccs, data_real_maccs]
        eval_combinations(
            names=[names_prefixes + x for x in names_maccs],
            datas=datas_maccs,
            shape=shape_maccs,
            metric=CompiledMetric(
                pointer=distances_numba[0],
                kind=MetricKind.Tanimoto,
                signature=MetricSignature.ArrayArray,
            )
            if use_numba
            else distance_simsimd,
            batch_size=batch_size,
        )

        # ECFP4 fingerprints
        data_pubchem_ecfp4 = chunks_pubchem.head(
            max_molecules, shape_ecfp4, shuffle=True
        )
        data_gdb13_ecfp4 = chunks_gdb13.head(max_molecules, shape_ecfp4, shuffle=True)
        data_real_ecfp4 = chunks_real.head(max_molecules, shape_ecfp4, shuffle=True)

        datas_ecfp4 = [data_pubchem_ecfp4, data_gdb13_ecfp4, data_real_ecfp4]
        eval_combinations(
            names=[names_prefixes + x for x in names_ecfp4],
            datas=datas_ecfp4,
            shape=shape_ecfp4,
            metric=CompiledMetric(
                pointer=distances_numba[1],
                kind=MetricKind.Tanimoto,
                signature=MetricSignature.ArrayArraySize,
            )
            if use_numba
            else distance_simsimd,
            batch_size=batch_size,
        )

        # Mixed MACCS+ECFP4 fingerprints
        data_pubchem_mixed = chunks_pubchem.head(
            max_molecules, shape_mixed, shuffle=True
        )
        data_gdb13_mixed = chunks_gdb13.head(max_molecules, shape_mixed, shuffle=True)
        data_real_mixed = chunks_real.head(max_molecules, shape_mixed, shuffle=True)

        datas_mixed = [data_pubchem_mixed, data_gdb13_mixed, data_real_mixed]
        eval_combinations(
            names=[names_prefixes + x for x in names_mixed],
            datas=datas_mixed,
            shape=shape_mixed,
            metric=CompiledMetric(
                pointer=distances_numba[2],
                kind=MetricKind.Tanimoto,
                signature=MetricSignature.ArrayArraySize,
            )
            if use_numba
            else distance_simsimd,
            batch_size=batch_size,
        )

        # Conditional MACCS*ECFP4 fingerprints
        eval_combinations(
            names=[names_prefixes + x for x in names_conditional],
            datas=datas_mixed,
            shape=shape_mixed,
            metric=CompiledMetric(
                pointer=distances_numba[3],
                kind=MetricKind.Tanimoto,
                signature=MetricSignature.ArrayArray,
            ),
            batch_size=batch_size,
        )
