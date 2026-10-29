"""Deterministic semantic chunk selection."""
from collections import defaultdict
import numpy as np
from keychunk.budget import DEFAULT_RECOMPUTE_CHUNKS,validate_recompute_chunks
from .types import ClusterPlan


def build_cluster_plan(vectors,threshold=.5,*,skip_clustering=False):
    # Optional helper: when retrieved documents contain redundant content, cluster
    # their chunks first. The centered cosine-distance threshold defaults to 0.5
    # and can be adjusted to suit the documents. If retrieval already includes
    # deduplication or redundancy filtering, skip this step with
    # skip_clustering=True (--skip-clustering).
    if not isinstance(skip_clustering,bool):raise ValueError('skip_clustering must be a boolean')
    values=np.asarray(vectors,dtype=np.float64)
    if values.ndim!=2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError('semantic vectors must be a finite nonempty matrix')
    if not 0.<threshold<=2.:raise ValueError('distance threshold must be in (0,2]')
    centered=values-values.mean(axis=0,keepdims=True)
    norms=np.linalg.norm(centered,axis=1,keepdims=True)
    normalized=centered/np.maximum(norms,1e-12)
    similarity=np.clip(normalized@normalized.T,-1.,1.)
    if skip_clustering:labels=range(len(values))
    elif len(values)==1:labels=(0,)
    else:
        from sklearn.cluster import AgglomerativeClustering
        distance=np.clip(1.-similarity,0.,2.);np.fill_diagonal(distance,0.)
        labels=AgglomerativeClustering(n_clusters=None,metric='precomputed',
            linkage='complete',distance_threshold=threshold).fit_predict(distance)
    grouped=defaultdict(list)
    for index,label in enumerate(labels):grouped[int(label)].append(index)
    groups=tuple(tuple(grouped[label]) for label in sorted(grouped))
    representatives=tuple(sorted(max(group,key=lambda i:(
        sum(float(similarity[i,j]) for j in group if j!=i),-i)) for group in groups))
    return ClusterPlan(representatives,groups,similarity)


def _greedy(similarity,candidates,count):
    remaining=list(candidates);chosen=[]
    while remaining and len(chosen)<count:
        def score(index):
            positive=[float(similarity[index,j]) for j in remaining
                if j!=index and similarity[index,j]>0.]
            return len(positive),abs(sum(positive)),-index
        best=max(remaining,key=score);chosen.append(best);remaining.remove(best)
    return chosen


def select_from_plan(plan,count=DEFAULT_RECOMPUTE_CHUNKS):
    n=len(plan.similarity);count=validate_recompute_chunks(count,n)
    chosen=_greedy(plan.similarity,plan.representatives,count)
    if len(chosen)<count:
        chosen+=_greedy(plan.similarity,[i for i in range(n) if i not in chosen],count-len(chosen))
    return tuple(chosen)


def select_chunks(vectors,recompute_chunks=DEFAULT_RECOMPUTE_CHUNKS,*,threshold=.5,skip_clustering=False):
    """Return exactly the requested number of original chunk indices."""
    return select_from_plan(build_cluster_plan(vectors,threshold,skip_clustering=skip_clustering),recompute_chunks)
