import math
class ClusterNode:
    def __init__(self,center):
        self.center = center

class Hierarchical():
    def __init__(self, data):
        self.data = data
        self.dist_cache = dict()

    def eucli_distance(self, col1, col2):
        if (col1, col2) in self.dist_cache:
            return self.dist_cache[(col1, col2)]
        _dist = self.data.loc[col1, col2].mean().mean()
        self.dist_cache[(col1, col2)] = _dist
        return _dist

    def traverse(self,node):
        return node.center

    def hcluster(self, col_list, ruleD=0.8, ruleN=2):
        ruleD = float(ruleD)
        ruleN = int(float(ruleN))
        nodes=[ClusterNode(center=v) for i,v in enumerate(col_list)]
        distances = {}
        currentclustid = -1
        while len(nodes) > ruleN:
            min_dist=math.inf
            nodes_len = len(nodes)
            closest_part = None
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    d_key = (nodes[i].center, nodes[j].center)
                    if d_key not in distances:
                        distances[d_key] = self.eucli_distance(
                            nodes[i].center,
                            nodes[j].center)
                    d = distances[d_key]
                    if d < min_dist and d<= ruleD:
                        min_dist = d
                        closest_part = (i, j)
            if closest_part is None:
                break
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_center = nodes[part1].center + nodes[part2].center
            new_node = ClusterNode(center=new_center)
            currentclustid -= 1
            del nodes[part2], nodes[part1]
            nodes.append(new_node)
        self.nodes = nodes
        return [self.traverse(nodes[i]) for i in range(len(nodes))]


class col_cluster(Hierarchical):
    @staticmethod
    def from_model(m):
        o = col_cluster(1 - (m.corr ** 2))
        o.m = m
        o.entL = o.m.entL
        return o

    @staticmethod
    def from_data(data, entL):
        o = col_cluster(1 - data ** 2)
        o.entL = entL
        return o

    def sort(self, cluster_res):
        cluster_res1 = [self.entL.loc[list(i)] for i in cluster_res]
        cluster_res1.sort(key=lambda x:x.max(), reverse=True)
        return cluster_res1

    def cluster(self, cols, ruleD=0.8, ruleN=2):
        col_list = [(i, ) for i in cols]
        res = self.hcluster(col_list=col_list,
                            ruleD=ruleD,
                            ruleN=ruleN)
        return self.sort(res)

if __name__ == "test":
    col_cluster.from_model(sb).cluster(sb.corr.index.tolist())[0]
    cols = [(i, ) for i in self.corr.index]
    DISTANCE = (1-(self.corr)**2)
    h = Hierarchical(data=DISTANCE)
    sg = h.hcluster(col_list=cols, ruleD=0.95, ruleN=3)
