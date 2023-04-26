import faiss

class InferenceModel:
  def __init__(self, name, model, sentences, verbose=False):
    self.name = name
    self.model = model 
    self.sentences = sentences

    # create the index
    # Step 1 embed sentences
    if verbose:
      print(f"embedding {len(sentences)} sentences using the pretrained embedding from the passed model")
    self.embeddings = self.embed(sentences, show_progress_bar=verbose)

    # Step 2 create the underlying index structure
    if verbose:
      print(f"creating an index of the embeddings")
    self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
    self.index.add(self.embeddings)

  def __str__(self):
    return f"{self.name}: {len(self.sentences)} embedded sentences."

  def __repr__(self):
    return str(self)

  def embed(self, sents, show_progress_bar=False):
    return self.model.encode(sents, show_progress_bar=show_progress_bar)

  def query(self, sents, k=16, include_dist=False):
    lst = [sents] if isinstance(sents, str) else sents
    vec = self.embed(lst)

    # retrieve indices of k+1 closest sentences
    dists, indices = self.index.search(vec, k=k+1)
    # for each sentence, only keep the top k ones that are NOT the sentence itself 
    sents = [[self.sentences[idx] for idx in row if self.sentences[idx] != lst[row_idx]][:k] for row_idx, row in enumerate(indices)]

    if len(sents) == 1:
      dists = dists[0]
      sents = sents[0]

    if include_dist:
      return sents, dists
    
    return sents