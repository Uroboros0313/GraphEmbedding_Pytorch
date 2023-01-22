import random

from tqdm import tqdm

            
class RandomWalker():
    def __init__(
        self,
        G,
        walk_len,
        num_walk,
        weighted_walk,
        ) -> None:
        
        self.G = G
        self.walk_len = walk_len
        self.num_walk = num_walk
        self.weighted_walk = weighted_walk
        
    def gen_dw_walks(self):
        nodes = list(self.G.nodes)
        walks = []
        for node in tqdm(nodes):
            walks.extend(self.__gen_node_walk(node))
            
        return walks

    def __gen_node_walk(self, node):
        node_walks = []
        for i in range(self.num_walk): 
            walk = [node]
            cur = node
            for j in range(self.walk_len):
                neighs = list(self.G.neighbors(cur))
                if len(neighs) <= 0:
                    break

                # weighted walk
                if self.weighted_walk != 0:
                    weights = [self.G[cur][neigh]['weight'] for neigh in neighs]
                    next = random.choices(neighs, weights=weights, k=1)[0]
                    walk.append(next)
                    cur = next
                
                else:
                    next = random.choice(neighs)
                    walk.append(next)
                    cur = next
            node_walks.append(walk)
            
        return node_walks
                
        