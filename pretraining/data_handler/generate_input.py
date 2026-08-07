import glob
from collections import Counter
import os
import pandas as pd
import numpy as np
import random
import networkx as nx
import torch
from torch_geometric.data import Data    
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle
from sklearn.model_selection import train_test_split


def _get_embedding_column(df, ppi_feat_dir):
    preferred_columns = []
    feat_path = str(ppi_feat_dir).lower()

    if "prostt5" in feat_path or "prosst5" in feat_path:
        preferred_columns.append("ProstT5-Embeddings")
    if "esm" in feat_path:
        preferred_columns.append("ESM2-Embeddings")

    preferred_columns.extend(["ProstT5-Embeddings", "ESM2-Embeddings"])
    seen = set()

    for column in preferred_columns:
        if column in seen:
            continue
        seen.add(column)
        if column in df.columns:
            return column

    embedding_columns = [col for col in df.columns if col.endswith("-Embeddings")]
    if len(embedding_columns) == 1:
        return embedding_columns[0]

    raise KeyError(
        f"Could not determine embedding column for {ppi_feat_dir}. "
        f"Available embedding columns: {embedding_columns}"
    )


def split_data_context(num_y):
    
    split_idx = list(range(num_y))
    random.shuffle(split_idx)
    train_idx = split_idx[ : int(len(split_idx) * 0.8)] # Train mask
    train_mask = torch.zeros(num_y, dtype = torch.bool)
    train_mask[train_idx] = 1
    val_idx = split_idx[ int(len(split_idx) * 0.8) : int(len(split_idx) * 0.9)] # Val mask
    val_mask = torch.zeros(num_y, dtype=torch.bool)
    val_mask[val_idx] = 1
    test_idx = split_idx[int(len(split_idx) * 0.9) : ] # Test mask
    test_mask = torch.zeros(num_y, dtype=torch.bool)
    test_mask[test_idx] = 1
    
    return train_mask, val_mask, test_mask


def split_data_global(
    edges,
    edges_to_sets,
    dict_edge_count=None,
    verbose=False):
    train_mask = torch.zeros(len(edges), dtype = torch.bool)
    val_mask = torch.zeros(len(edges), dtype = torch.bool)
    test_mask = torch.zeros(len(edges), dtype = torch.bool)
    
    if not dict_edge_count is None:
        edge_weights = torch.zeros(len(edges), dtype = torch.float32)
    
    for i, (x, y) in enumerate(edges):
        
        try:
            tuple_ = (x, y)
            set_ = edges_to_sets[tuple_]
        except:
            tuple_ = (y, x)
            set_ = edges_to_sets[tuple_]
        if set_ == 'train':
            train_mask[i] = 1
        elif set_ == 'val':
            val_mask[i] = 1
        elif set_ == 'test':
            test_mask[i] = 1

        if not dict_edge_count is None:
            edge_weights[i] = 1. / dict_edge_count[tuple_]
            
    if verbose:
        print('train_mask:', torch.unique(train_mask, return_counts=True))
        print('val_mask:', torch.unique(val_mask, return_counts=True))
        print('test_mask:', torch.unique(test_mask, return_counts=True))
    
    if dict_edge_count is None:
        return train_mask, val_mask, test_mask
    else:
        return train_mask, val_mask, test_mask, edge_weights
             
    
def compute_degree(
    ppi_dir, G, dataset_mode, verbose):
    
    dict_degree = {node: 0 for node in G.nodes}
    
    for i_f, f in tqdm(enumerate(glob.glob(os.path.join(ppi_dir, "*.txt"))), desc='browsing PPI files to compute global degree'):
        # Read edgelist
        ppi = nx.read_edgelist(f)
        ppi_nodes_ = list(ppi.nodes())
        ppi_edges_ = list(ppi.edges())
        
        for edge in list(ppi.edges):
            x, y = edge
            dict_degree[x] += 1
            dict_degree[y] += 1
        print('--- f:', f)
        print('current degree counts:', np.unique(list(dict_degree.values()), return_count=True))
    return dict_degree


def compute_count_edge(
    ppi_dir, G, dataset_mode, verbose):
    
    """
    # make G symmetric
    print('# G edges:', len(G.edges))
    G.add_edges_from([(y,x) for x,y in list(G.edges)])
    print('# updated G edges:', len(G.edges))
    """
    
    dict_edges = {}
    for edge in tqdm(list(G.edges), desc='browse edges to compute global edges count'):
        dict_edges[edge] = 0
    
    for i_f, f in tqdm(enumerate(glob.glob(os.path.join(ppi_dir, "*.txt"))), desc='counting edges across PPI files'): # Expected format of filename: <PPI_DIR>/<CONTEXT>.<suffix>

        # Parse name of context
        context = os.path.splitext(os.path.basename(f))[0]
        
        if dataset_mode != "bulk":
            context = context.replace("_", " ")
        
        # Read edgelist
        ppi = nx.read_edgelist(f)
        
        for edge in list(ppi.edges):
            try:
                dict_edges[edge] += 1
            except:
                x, y = edge
                dict_edges[(y,x)] += 1

    return dict_edges
        
        
def read_ppi(
    ppi_dir,
    removed_genes,
    global_G,
    verbose=False,
    dataset_mode="legacy",
    split_mode='context',
    count_edge_path=None,
    seed=0,
    weighted_ppi_loss=False):
    """
    Read PPI layers from the specified directory.

    Args:
        ppi_dir (str): Directory containing PPI layer files.
        removed_genes (list): Genes without protein embeddings to remove.
        global_G (nx.Graph): Global interactome used for global edge splits.
        verbose (bool): Print data-loading diagnostics.
        dataset_mode (str): Naming convention used by the PPI files.
        split_mode (str): Split edges independently by context or globally.
        count_edge_path (str, optional): Cached cross-context edge counts.
        seed (int): Random seed for global edge splitting.
        weighted_ppi_loss (bool): Return cross-context PPI edge weights.
        
    Returns:
        orig_ppi_layers (dict): Original PPI layers.
        ppi_layers (dict): PPI layers with relabeled nodes whose key corresponds to cell types.
        ppi_train (dict): Train masks for PPI layers.
        ppi_val (dict): Validation masks for PPI layers.
        ppi_test (dict): Test masks for PPI layers.
    """
    assert split_mode in ['context', 'global']
    
    orig_ppi_layers = dict()
    ppi_layers = dict()
    ppi_train = dict()
    ppi_val = dict()
    ppi_test = dict()
    ppi_weights = dict() # leveraged only if weighted_ppi_loss=True
        
    if split_mode == 'global':
        assert count_edge_path is not None
        if not os.path.exists(count_edge_path):
            # check at global_n_nodes without taking into account removed genes yet
            dict_edge_count = compute_count_edge(
                ppi_dir, global_G, dataset_mode, verbose)

            with open(count_edge_path, 'wb') as f:
                pickle.dump(dict_edge_count, f)
        else:
            with open(count_edge_path, 'rb') as f:
                dict_edge_count = pickle.load(f)
            
        # remove genes
        print('removed genes:', removed_genes)
        local_keys = list(dict_edge_count.keys())
        for key in local_keys:
            x, y = key
            if (x in removed_genes) or (y in removed_genes):
                del dict_edge_count[key]

        # identify stratified splits
        n_edges = len(list(dict_edge_count.keys()))
        idx_edges = np.arange(n_edges)
        edges = np.array(list(dict_edge_count.keys()))
        y = np.array(list(dict_edge_count.values())) # consider cumulated degrees as classes
        pos_degrees = np.argwhere(y != 0)[:, 0] # only keep edges with strictly positive degree
        idx_edges = idx_edges[pos_degrees]
        y = y[pos_degrees]
        print('unique degrees:', np.unique(y, return_counts=True))
        
        # binarize in 10 groups to avoid (class==degree) with only one sample
        y = (y - y.min()) * 10 / (y.max() - y.min())
        y = np.floor(y)
        y[y==10] = 9
        # compute splits
        idx_edges_train, idx_edges_test, y_train, y_test =  train_test_split(idx_edges, y, test_size=0.1, random_state=seed, stratify=y)
        idx_edges_subtrain, idx_edges_val, y_subtrain, y_val = train_test_split(idx_edges_train, y_train, test_size=1./9., random_state=seed, stratify=y_train)
        print('y_test:', np.unique(y_test, return_counts=True))
        print('y_subtrain:', np.unique(y_subtrain, return_counts=True))
        print('y_val:', np.unique(y_val, return_counts=True))
        edges_subtrain = edges[idx_edges_subtrain]
        edges_val = edges[idx_edges_val]
        edges_test = edges[idx_edges_test]
        
        edges_to_sets = {}
        for edges, s in [(edges_subtrain, 'train'), (edges_val, 'val'), (edges_test, 'test')]:
            for i in range(edges.shape[0]):
                x, y = edges[i]
                edge_tuple = (x, y)
                edges_to_sets[edge_tuple] = s
        
    for i_f, f in tqdm(enumerate(glob.glob(os.path.join(ppi_dir, "*.txt"))), desc='reading PPI files'): # Expected format of filename: <PPI_DIR>/<CONTEXT>.<suffix>

        # Parse name of context
        context = os.path.splitext(os.path.basename(f))[0]
        
        if dataset_mode != "bulk":
            context = context.replace("_", " ")
        
        # Read edgelist
        ppi = nx.read_edgelist(f)
        if verbose:
            print('ppi.nodes():', len(ppi.nodes()))
        for gene in removed_genes:
            try:
                ppi.remove_node(gene)
            except:
                continue
        if verbose:
            print('updated ppi.nodes():', len(ppi.nodes()))
        
        # Relabel PPI nodes
        mapping = {n: idx for idx, n in enumerate(ppi.nodes())}
        ppi_layers[context] = nx.relabel_nodes(ppi, mapping)
        orig_ppi_layers[context] = ppi
        
        if len(removed_genes) == 0:
            assert nx.is_connected(ppi_layers[context])
        else:
            if (not nx.is_connected(ppi_layers[context])) and verbose:
                print(f'WARNING: PPI graph for cell {context} is disconnected')
        # Split into train/val/test
        if split_mode == 'context':
            ppi_train[context], ppi_val[context], ppi_test[context] = split_data_context(len(ppi_layers[context].edges))
        else:
            if not weighted_ppi_loss:
                if verbose:
                    print('---- context:', context)
                ppi_train[context], ppi_val[context], ppi_test[context] = split_data_global(
                    ppi.edges, edges_to_sets, dict_edge_count=None, verbose=verbose)
            else:
                if verbose:
                    print('---- context:', context)
                ppi_train[context], ppi_val[context], ppi_test[context], ppi_weights[context] = split_data_global(
                    ppi.edges, edges_to_sets, dict_edge_count=dict_edge_count, verbose=verbose)
            
    if verbose:
        print('orig_ppi_layers:', list(orig_ppi_layers.keys())[:10])
        print('ppi_layers:', list(ppi_layers.keys())[:10])
        print('ppi_train:', list(ppi_train.keys())[:10])
        print('ppi_val:', list(ppi_val.keys())[:10])
        print('ppi_test:', list(ppi_test.keys())[:10])
    
    return orig_ppi_layers, ppi_layers, ppi_train, ppi_val, ppi_test, ppi_weights


def create_data(G, train_mask, val_mask, test_mask, node_type, edge_type, x, symmetric=False, verbose=False,
                ppi_weights=None):
    """
    Create a PyTorch Geometric Data object from a NetworkX graph.
    Args:
        G (nx.Graph): NetworkX graph object.
        train_mask (torch.Tensor): Boolean mask for training edges.
        val_mask (torch.Tensor): Boolean mask for validation edges.
        test_mask (torch.Tensor): Boolean mask for test edges.
        node_type (list): List of node types.
        edge_type (list): List of edge types.
        x (torch.Tensor): Node feature matrix.
    Returns:
        new_G (Data): PyTorch Geometric Data object."""
    
    if not symmetric:
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        y = torch.ones(edge_index.size(1))
        num_classes = len(torch.unique(y))
        node_type = torch.tensor(node_type)
        edge_type = torch.tensor(edge_type)
        
        if not ppi_weights is None:
            ppi_weights_ = ppi_weights
    
    else:
        # Make sure that the graph is undirected
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        if verbose:
            print('(init)  edge_index:', edge_index.shape)
        
        sym_edge_index = torch.zeros_like(edge_index)
        sym_edge_index[0, :] = edge_index[1, :]
        sym_edge_index[1, :] = edge_index[0, :]
        edge_index = torch.cat([edge_index, sym_edge_index], dim=1)
        # Do not deduplicate here because it would misalign the edge masks.
        if verbose:
            print('(usym) edge_index:', edge_index.shape)
        y = torch.ones(edge_index.size(1))
        num_classes = len(torch.unique(y))
        node_type = torch.tensor(node_type)
        edge_type = torch.tensor([edge_type[0]] * edge_index.shape[1])
        if verbose:
            print('train_maks:', train_mask.shape, 'val_mask:', val_mask.shape, 'test_mask:', test_mask.shape)
            print('edge_type:', edge_type.shape)
        train_mask = torch.concat([train_mask, train_mask], dim=0) # Duplicate the train mask
        val_mask = torch.concat([val_mask, val_mask], dim=0) # Duplicate the val mask
        test_mask = torch.concat([test_mask, test_mask], dim=0) # Duplicate the test mask
        
        if not ppi_weights is None:
            ppi_weights_ = torch.concat([ppi_weights, ppi_weights], dim=0) # Duplicate ppi_weights
    
    # Create Data object
    new_G = Data(
            x = x,
            y = y,
            num_classes = num_classes,
            edge_index = edge_index,
            node_type = node_type,
            edge_attr = edge_type,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask
        )
    if not ppi_weights is None:
        new_G.ppi_weights = ppi_weights_
    
    if symmetric:
        assert new_G.is_undirected()
    
    return new_G


def read_global_ppi(f):
    # Read table from csv file
    graph_df = pd.read_csv(f)

    # Create a list of tuples, where each tuple is an edge
    edges = [(s, t) for s, t in zip(graph_df["protein1"].tolist(), graph_df["protein2"].tolist())]

    # Instantiate graph object
    G = nx.Graph()

    # Add edges (from the table) to the graph object
    G.add_edges_from(edges)

    return G

def compute_CT_map(ppi_data, tissue_neighbors, verbose=True):
    """
    Compute the mapping of cell types to tissues.
    
    Args:
    ppi_data (dict): Dictionary of PPI data objects for each cell type.
    tissue_neighbors (dict): Dictionary mapping tissue nodes to their neighbors.
    
    Returns:
    mapping_cell_to_tissues (dict): Mapping of cell types to tissues.
    Each key is a cell type, and the value is a dictionary with:
        - 'tissue_ids': List of tissue IDs associated with the cell type.
        - 'onehot': One-hot encoded tensor representing the tissue assignments.
    """
    
    # Create a mapping of cell types to tissues 
    list_cells = list(ppi_data.keys())
    max_cell_id = np.max(list_cells)
    n_cells = len(list_cells)
    
    n_tissues = len(tissue_neighbors.keys())
    unique_tissues = np.unique(list(tissue_neighbors.keys()))
    
    if verbose:
        print('unique_tissues:', unique_tissues)
    mapping_cell_to_tissues = {}
    for cell_key in ppi_data.keys():
        mapping_cell_to_tissues[cell_key] = {'tissue_ids':[], 'onehot':torch.zeros(n_tissues, dtype=torch.int32)} # Initialize empty list for each cell type
    
    for id_key, tissue_key in tqdm(enumerate(unique_tissues), desc='Computing cell to tissue mapping', disable=not verbose):
        for cell_key in tissue_neighbors[tissue_key]:
            if cell_key <= max_cell_id: # tissue_neighbors can also contain TT links
                mapping_cell_to_tissues[cell_key]['tissue_ids'].append(tissue_key)
                mapping_cell_to_tissues[cell_key]['onehot'][id_key] = 1
    
    return mapping_cell_to_tissues
    
def read_data(
    G_f, ppi_dir, mg_f, feat_mat_dim=None, get_CT_map=False,
    ppi_feat_dir=None,
    verbose=True,
    symmetric_ppi=False,
    run_pca=False,
    dataset_mode="legacy",
    split_mode='context',
    count_edge_path=None,
    weighted_ppi_loss=False,
    defer_ppi_features=False):
    """Load the global PPI, context-specific PPIs, and metagraph.

    ``ppi_feat_dir`` may provide pretrained protein features; otherwise random
    features of size ``feat_mat_dim`` are used. Edge splits can be defined per
    context or once on the global interactome. When requested, ``CT_map`` stores
    each cell type's tissue assignments as a one-hot vector.

    Returns:
        Tuple containing PPI data, metagraph data, edge-type indices, cell-type
        indices, tissue neighbors, original PPI layers, the original metagraph,
        and optionally the cell-to-tissue map.
    """

    # Read global PPI network
    G = nx.read_edgelist(G_f)
    
    # Feature Matrix for All Proteins
    removed_genes = []
    
    if ppi_feat_dir is None:
        print('set random features as initial protein node features')
        feat_mat = torch.normal(torch.zeros(len(G.nodes), feat_mat_dim), std=1)

    else:
        print(f'set protein embeddings as initial protein node features from {ppi_feat_dir}')
        
        # Read df that contains at least 'gene_name'
        df = pd.read_pickle(ppi_feat_dir)
        embedding_col = _get_embedding_column(df, ppi_feat_dir)
        print(f'using embedding column: {embedding_col}')
        feat_list = []
        
        # gene : encoded via gene name - not number in PPI networks
        for gene in tqdm(G.nodes, desc='recover protein embeddings', disable= not verbose): 
            embeddings = df.loc[df['gene_name'] == gene, embedding_col].values
            if embeddings.shape[0] > 0: # list of one numpy element
                embeddings = torch.tensor(embeddings[0])
            else: 
                removed_genes.append(gene)
                continue
            feat_list.append(embeddings)
        # Convert it to a tensor
        feat_mat = torch.stack(feat_list)
        print('--- missing genes:', len(removed_genes))
        # need to do something cleaner for data normalization later on
        feat_mat_mean = feat_mat.mean(dim=0)
        feat_mat_std = feat_mat.std(dim=0)
        
        feat_mat = (feat_mat - feat_mat_mean) / feat_mat_std
        print('feat_mat:', feat_mat.shape)
        if run_pca:
            # we identify whether all embedding dimensions are used for this set of proteins
            print('running PCA')
            pca = PCA(n_components=1500, random_state=0, svd_solver='arpack').fit(feat_mat.numpy())
            print('PCA ev: ', pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_))
            
    # Read PPI layers
    
    ## ppi_weights = empty dict if weighted_ppi_loss=False
    ## else ppi_weights[context] contains weights to put in the loss for ppi predictions in the context
    orig_ppi_layers, ppi_layers, ppi_train, ppi_val, ppi_test, ppi_weights = read_ppi(
        ppi_dir, removed_genes,
        G, dataset_mode=dataset_mode,
        split_mode=split_mode,
        count_edge_path=count_edge_path,
        weighted_ppi_loss=weighted_ppi_loss)
    print(f"Number of PPI layers: orig = {len(orig_ppi_layers)} / ppi_layers = {len(ppi_layers)} / train = {len(ppi_train)}/ val = {len(ppi_val)} / test = {len(ppi_test)}")
    
    # remove genes with missing embeddings from global ppi
    if len(removed_genes) > 0:
        for gene in removed_genes:
            G.remove_node(gene)
    
    # PPI layers are keyed by cell type. The original layers retain gene names;
    # the relabeled layers use independent integer indices in each context.
    
    # Read metagraph
    metagraph = nx.read_edgelist(mg_f, data=False, delimiter = "\t", create_using=nx.DiGraph)
    print('len(metagraph.nodes):', len(metagraph.nodes))
    # Make sure that is is connected 
    assert nx.is_connected(metagraph.to_undirected())
    
    # metagraph feature matrix 
    mg_feat_mat = torch.zeros(len(metagraph.nodes), feat_mat.shape[-1])
    
    print('------------------------------------------------------------------')
    print(f"metagraph - len = {len(metagraph)} / type = {type(metagraph)}")
    print('------------------------------------------------------------------')

    orig_mg = metagraph
    print("[METAGRAPH] Number of nodes:", len(metagraph.nodes), "Number of edges:", len(metagraph.edges))
    
    
    # mg_mapping contains each cell and tissues as keys
    metagraph_tissues = sorted([n for n in metagraph.nodes if "BTO" in n])
    metagraph_cells = sorted([n for n in metagraph.nodes if "BTO" not in n])

    cell_nodes = []
    # include PPI cell nodes that exist in metagraph (preserve original order as much as possible)
    for n in sorted(ppi_layers):
        if (dataset_mode == "bulk" and n in metagraph.nodes) or dataset_mode != "bulk":
            cell_nodes.append(n)
    # add any remaining metagraph cell nodes that were not already included
    for n in metagraph_cells:
        if n not in cell_nodes:
            cell_nodes.append(n)

    ordered_nodes = cell_nodes + metagraph_tissues
    mg_mapping = {n: i for i, n in enumerate(ordered_nodes)}
    if verbose:
        print('mg_mapping:', len(mg_mapping))
    
    assert len(mg_mapping) == len(metagraph.nodes), set(metagraph.nodes).difference(set(list(mg_mapping.keys())))
    
    # Set up Data object
    mg_nodetype = [0 if "BTO" in n else 1 for n in mg_mapping] # Tissue nodes = 0, Cell-type nodes = 1
    mg_edgetype = []
    
    for edges in tqdm(metagraph.edges, desc='Creating metagraph edge types'):
        if "BTO" in edges[0] and "BTO" in edges[1]: mg_edgetype.append(0) # tissue-tissue edge
        elif "BTO" in edges[0] and "BTO" not in edges[1]: mg_edgetype.append(1) # tissue-cell edge
        elif "BTO" not in edges[0] and "BTO" in edges[1]: mg_edgetype.append(2) # cell-tissue edge
        elif "BTO" not in edges[0] and "BTO" not in edges[1]: mg_edgetype.append(3) # cell-cell edge
        else:
            print(edges)
            raise NotImplementedError
        
    # Map every tissue node to its neighboring tissues and cell types.
    tissue_neighbors = {mg_mapping[t]: [mg_mapping[n] for n in metagraph.to_undirected().neighbors(t)] for t in metagraph.to_undirected().nodes if "BTO" in t}
    
    metagraph = nx.relabel_nodes(metagraph, mg_mapping)
    mg_mask = torch.ones(len(metagraph.edges), dtype = torch.bool) # Pass in all meta graph edges during training, validation, and test
    mg_data = create_data(
        metagraph, mg_mask, mg_mask, mg_mask, mg_nodetype, mg_edgetype, mg_feat_mat)

    # Set up PPI Data objects
    orig_ppi_layers_remapped = {mg_mapping[k]: v for k, v in orig_ppi_layers.items() if k in mg_mapping}
    ppi_layers = {mg_mapping[k]: v for k, v in ppi_layers.items() if k in mg_mapping}
    ppi_train = {mg_mapping[k]: v for k, v in ppi_train.items() if k in mg_mapping}
    ppi_val = {mg_mapping[k]: v for k, v in ppi_val.items() if k in mg_mapping}
    ppi_test = {mg_mapping[k]: v for k, v in ppi_test.items() if k in mg_mapping}
    if weighted_ppi_loss:
        ppi_weights = {mg_mapping[k]: v for k, v in ppi_weights.items() if k in mg_mapping}
        
    ppi_data = dict()
    global_node_index = {p: idx for idx, p in enumerate(list(G.nodes()))}
    
    for cluster, ppi in tqdm(ppi_layers.items(), desc='Creating PPI data objects'):
        ppi_nodetype = [2] * len(ppi.nodes) # protein nodes = 2
        if not symmetric_ppi:
            ppi_edgetype = [4] * len(ppi.edges) # protein-protein edge
        else:
            ppi_edgetype = [4] * (2 * len(ppi.edges)) # protein-protein edge
        
        ppi_node_names = list(orig_ppi_layers_remapped[cluster].nodes())
        p_index = [global_node_index[p] for p in ppi_node_names]
        assert len(p_index) == len(ppi_node_names)
        if defer_ppi_features:
            c_feat_mat = feat_mat.new_empty((len(p_index), 0))
        else:
            c_feat_mat = feat_mat[p_index, :]
        assert c_feat_mat.shape[0] == len(ppi.nodes)
        
        if weighted_ppi_loss:
            ppi_data[cluster] = create_data(
                ppi, ppi_train[cluster], ppi_val[cluster], ppi_test[cluster],
                ppi_nodetype, ppi_edgetype, c_feat_mat, symmetric=symmetric_ppi,
                ppi_weights=ppi_weights[cluster])
        else:
            ppi_data[cluster] = create_data(
                ppi, ppi_train[cluster], ppi_val[cluster], ppi_test[cluster],
                ppi_nodetype, ppi_edgetype, c_feat_mat, symmetric=symmetric_ppi)
        if defer_ppi_features:
            ppi_data[cluster].feature_index = torch.as_tensor(p_index, dtype=torch.long)

    if defer_ppi_features:
        mg_data.global_protein_features = feat_mat
        
            
    #  Set up edge attr dict
    edge_attr_dict = {"tissue_tissue": 0, "tissue_cell": 1, "cell_tissue": 2, "cell_cell": 3, "protein_protein": 4}
    
    # Return celltype specific PPI network data
    
    if get_CT_map:
        # Compute cell to tissues mapping
        CT_map = compute_CT_map(ppi_data, tissue_neighbors)

        return ppi_data, mg_data, edge_attr_dict, mg_mapping, tissue_neighbors, orig_ppi_layers, orig_mg, CT_map
    else:
        return ppi_data, mg_data, edge_attr_dict, mg_mapping, tissue_neighbors, orig_ppi_layers, orig_mg


def subset_ppi(num_subset, ppi_data, ppi_layers):
    
    # Take a subset of PPI data objects
    new_ppi_data = dict()
    for celltype, ppi in ppi_data.items():
        if len(new_ppi_data) < num_subset:
            new_ppi_data[celltype] = ppi
    ppi_data = new_ppi_data

    # Take a subset of PPI layers
    new_ppi_layers = dict()
    for celltype, ppi in ppi_layers.items():
        if len(new_ppi_layers) < num_subset:
            new_ppi_layers[celltype] = ppi_layers[celltype]
    ppi_layers = new_ppi_layers

    return ppi_data, ppi_layers


def get_metapaths():
    ppi_metapaths = [[4]] # Get PPI metapaths
    mg_metapaths = [[0], [1], [2], [3]]
    return ppi_metapaths, mg_metapaths


def get_centerloss_labels(celltype_map, ppi_layers, verbose=False):
    """
    Generate center loss labels and masks for training, validation, and testing.
    Args:
        celltype_map (dict): Mapping of cell types to indices.
        ppi_layers (dict): PPI layers with relabeled nodes whose key corresponds to cell types.
    Returns:
        center_loss_labels (list): List of center loss labels corresponding to each PPI node.
        train_mask (list): List of indices for training nodes.
        val_mask (list): List of indices for validation nodes.
        test_mask (list): List of indices for testing nodes.
    """
    center_loss_labels = []
    
    if verbose:
        print('celltype_map:', list(celltype_map.keys())[:50], len(celltype_map))
        print('ppi_layers:', list(ppi_layers.keys())[:50], len(ppi_layers))
    for celltype, ppi in ppi_layers.items():
        center_loss_labels += [celltype_map[celltype]] * len(ppi.nodes)
        
    center_loss_idx = random.sample(range(len(center_loss_labels)), len(center_loss_labels))
    train_mask = center_loss_idx[ : int(0.8 * len(center_loss_idx))]
    val_mask = center_loss_idx[len(train_mask) : len(train_mask) + int(0.1 * len(center_loss_idx))]
    test_mask = center_loss_idx[len(train_mask) + len(val_mask) : ]
    
    if verbose:
        print("Center loss labels (cell_id, number of proteins):", Counter(center_loss_labels))
        print(f'train_mask: {len(train_mask)} / val_mask: {len(val_mask)} / test_mask: {len(test_mask)}')

    return center_loss_labels, train_mask, val_mask, test_mask


def graph_saint_dryrun(
    data,
    save_dir,
    batch_size,
    num_steps=16,
    sample_coverage=1000, # to estimate sampling stats / rule : while total_sampled_nodes < self.N * self.sample_coverage:
    num_workers=0,
    pin_memory=True):
    
    _ = GraphSAINTEdgeSampler(
            data,
            batch_size = batch_size,
            num_steps = num_steps,
            log=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            save_dir=save_dir,
            sample_coverage=sample_coverage
            )
