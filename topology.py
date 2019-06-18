import random
import argparse


def get_graph(n_nodes, topology):
        n_nodes = int(n_nodes)
        nodes = [i for i in range(n_nodes)]
        if topology == "full":
                graph = {i: nodes[:] for i in nodes}
        elif topology == "ring":
                graph = {i: [] for i in nodes}
                for i in graph:
                        if i-1 in nodes:
                                graph[i].append(i-1)
                        else:
                                graph[i].append(nodes[-1])
                        if i+1 in nodes:
                                graph[i].append(i+1)
                        else:
                                graph[i].append(nodes[0])
        elif topology == "random":
                visited = []
                not_visited = nodes[:]
                graph = {}
                first_node = random.choice(nodes)
                visited.append(first_node)
                not_visited.remove(first_node)
                while not_visited:
                        previous_node = random.choice(visited)
                        next_node = random.choice(not_visited)
                        try:
                                graph[previous_node].append(next_node)
                        except:
                                graph[previous_node] = [next_node]
                        visited.append(next_node)
                        not_visited.remove(next_node)

#               n_rand_edges = random.randint(

        for node in graph:
                try:
                        graph[node].remove(node)
                except ValueError:
                        pass
        return graph

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Integer indicating the graph degree & string indicating topology")
        parser.add_argument("-n_nodes", help="Number of nodes in graph", default=5, required=True)
        parser.add_argument("-topology", help="'full', 'ring' or 'random'", default='random', required=True)
        args = parser.parse_args()

        print(get_graph(args.n_nodes, args.topology))

