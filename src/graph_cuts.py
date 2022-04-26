import maxflow
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse
from copy import copy


class Image_blending:
    """
    Main class for image synthesis with graph cuts
    """
    def __init__(self, Source, sink, Mask, construct_graph=False):
        """
        Initializes the graph and computes the min-cut.
        :param Source: image to be blended (foreground)
        :param sink: background image
        :param Mask: manual Mask with constrained pixels
        :param construct_graph: if true, graph is saved
        """
        assert (Source.shape == sink.shape),"Source and sink must be of same dimensions"

        # Creating a graph object from max-flow library
        graph = maxflow.Graph[float]()


        # adding the nodes to
        node_ids = graph.add_grid_nodes((Source.shape[0], Source.shape[1]))
        
        # function is called to compute edge weights
        self.edge_weights(Source, sink)

        # non-terminal edges are added 
        width = Source.shape[1]
        height = Source.shape[0]
        for i in range(height):
            for j in range(width):
                # computing the right neighbor
                if j + 1 < width:
                    weight = self.edge_weights[i, j, 0]
                    graph.add_edge(node_ids[i][j],node_ids[i][j + 1],weight,weight)

                #computing the bottom neighbor
                if i + 1 < height:
                    weight = self.edge_weights[i, j, 1]
                    graph.add_edge(node_ids[i][j], node_ids[i + 1][j], weight, weight)

                # Add terminal edge capacities for the pixels constrained to belong to the Source/sink.
                if np.array_equal(Mask[i, j, :], [0, 255, 255]):
                    graph.add_tedge(node_ids[i][j], 0, np.inf)
                elif np.array_equal(Mask[i, j, :], [255, 128, 0]):
                    graph.add_tedge(node_ids[i][j], np.inf, 0)

        # Plot graph
        if construct_graph:
            nxg = graph.get_nx_graph()
            #self.plot_graph_2d(nxg, height, width)
            self.plot_graph_2d(graph, node_ids.shape)

        # Compute max flow / min cut.
        flow = graph.maxflow()
        print(flow)
        self.sgm = graph.get_grid_segments(node_ids)
        self.sgm1 = self.sgm

        # # Generating the intermediate output i.e marking the cut pixels on source with black colour
        src1 = copy(Source)
        src1[self.sgm1] = 0
        cv2.imwrite(os.path.join(image_dir,"sol_to_image_cut.png"), src1)

        # Plotting the pixels to be cut from the source
        img2 = np.int_(np.logical_not(self.sgm))
        plt.imshow(img2)
        plt.show()

    def edge_weights(self, Source, sink):
        """
        Computes edge weights based on matching quality cost.
        :param Source: image to be blended (foreground)
        :param sink: background image
        """
        self.edge_weights = np.zeros((Source.shape[0], Source.shape[1], 2))

        # generating shifted matrices for vector operations.
        source_left_shifted = np.roll(Source, -1, axis=1)
        sink_left_shifted = np.roll(sink, -1, axis=1)
        source_up_shifted = np.roll(Source, -1, axis=0)
        sink_up_shifted = np.roll(sink, -1, axis=0)

        # Assign edge weights.
        # For numerical stability, avoid divide by 0.
        eps = 1e-10

        # Right neighbor.
        weight = np.sum(np.square(Source - sink, dtype=np.float) + np.square(source_left_shifted - sink_left_shifted,  dtype=np.float), axis=2)
        normalizing_factor = np.sum(np.square(Source - source_left_shifted, dtype=np.float) +  np.square(sink - sink_left_shifted,  dtype=np.float), axis=2)
        self.edge_weights[:, :, 0] = weight / (normalizing_factor + eps)

        # Bottom neighbor.
        weight = np.sum(np.square(Source - sink, dtype=np.float) + np.square(source_up_shifted - sink_up_shifted, dtype=np.float), axis=2)
        normalizing_factor = np.sum(np.square(Source - source_up_shifted, dtype=np.float) + np.square(sink - sink_up_shifted,  dtype=np.float), axis=2)
        self.edge_weights[:, :, 1] = weight / (normalizing_factor + eps)

    def plot_graph_2d(self, graph, nodes_shape, plot_weights=False, plot_terminals=True, font_size=7):
        """
        Plot the graph to be used in graph cuts
        :param graph: PyMaxflow graph
        :param nodes_shape: patch shape
        :param plot_weights: if true, edge weights are shown
        :param plot_terminals: if true, the terminal nodes are shown
        :param font_size: text font size
        """
        X, Y = np.mgrid[:nodes_shape[0], :nodes_shape[1]]
        aux = np.array([Y.ravel(), X[::-1].ravel()]).T
        positions = {i: v for i, v in enumerate(aux)}
        positions['s'] = (-1, nodes_shape[0] / 2.0 - 0.5)
        positions['t'] = (nodes_shape[1], nodes_shape[0] / 2.0 - 0.5)

        graph_maxflow = graph.get_nx_graph()
        print("graph_maxflow created")
        if not plot_terminals:
            graph_maxflow.remove_nodes_from(['s', 't'])

        nx.draw(graph_maxflow, pos=positions)

        if plot_weights:
            edge_labels = {}
            for u, v, d in graph_maxflow.edges(data=True):
                edge_labels[(u, v)] = d['weight']
            nx.draw_networkx_edge_labels(graph_maxflow, pos=positions, edge_labels=edge_labels, label_pos=0.3, font_size=font_size)
        plt.axis('equal')
        #plt.show()
        plt.savefig("output.jpg")

    def blend(self, Source, target):
        """
        Blends the target image with the Source image based on the graph cut.
        :param Source: Source image
        :param target: Target image
        """
        target[self.sgm] = Source[self.sgm]
        return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='image_dir', required=True, help='Image directory')
    args = parser.parse_args()

    # Read the images and the Mask.
    image_dir = args.image_dir
    Source = cv2.imread(os.path.join(image_dir, 'src.jpg'))
    target = cv2.imread(os.path.join(image_dir, 'target.jpg'))
    Mask = cv2.imread(os.path.join(image_dir, 'mask.png'))

    # Compute the min-cut.
    graphcuts = Image_blending(Source, target, Mask)

    # Save the output.
    target = graphcuts.blend(Source, target)
    cv2.imwrite(os.path.join(image_dir, "result.png"), target)
