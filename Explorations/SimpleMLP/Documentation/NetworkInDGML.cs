using SimpleMLP.MLP;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Serialization;

namespace SimpleMLP.Documentation
{
    //https://stackoverflow.com/questions/8199600/c-sharp-directed-graph-generating-library
    public class NetworkInDGML
    {
        public static NetworkInDGML Create(Network network)
        {
            NetworkInDGML ret = new NetworkInDGML();

            List<Neuron> allNeurons = new List<Neuron>();
            allNeurons.AddRange(network.InputLayer.Neurons);
            foreach (HiddenLayer hiddenLayer in network.HiddenLayers)
            {
                allNeurons.AddRange(hiddenLayer.Neurons);
            }
            allNeurons.AddRange(network.OutputLayer.Neurons);

            foreach (Neuron neuron in allNeurons)
            {
                Node node = new Node();
                node.Id = neuron.UniqueName;
                node.Label = neuron.UniqueName;

                ret.AddNode(node);
            }

            foreach (Neuron neuron in allNeurons)
            {
                foreach (Dendrite dendrite in neuron.Dendrites)
                {
                    Neuron upStreamNeuron = dendrite.UpStreamNeuron;

                    Link link = new Link();
                    link.Label = "";
                    link.Source = upStreamNeuron.UniqueName;
                    link.Target = neuron.UniqueName;

                    ret.Links.Add(link);
                }
            }

            return ret;
        }

        public struct Graph
        {
            public Node[] Nodes;
            public Link[] Links;
        }

        public struct Node
        {
            [XmlAttribute]
            public string Id;
            [XmlAttribute]
            public string Label;

            public Node(string id, string label)
            {
                this.Id = id;
                this.Label = label;
            }
        }

        public struct Link
        {
            [XmlAttribute]
            public string Source;
            [XmlAttribute]
            public string Target;
            [XmlAttribute]
            public string Label;

            public Link(string source, string target, string label)
            {
                this.Source = source;
                this.Target = target;
                this.Label = label;
            }
        }

        protected List<Node> Nodes { get; private set; }
        protected List<Link> Links { get; private set; }

        private NetworkInDGML()
        {
            Nodes = new List<Node>();
            Links = new List<Link>();
        }

        private void AddNode(Node n)
        {
            this.Nodes.Add(n);
        }

        private void AddLink(Link l)
        {
            this.Links.Add(l);
        }

        public void Serialize(string xmlpath)
        {
            Graph g = new Graph();
            g.Nodes = this.Nodes.ToArray();
            g.Links = this.Links.ToArray();

            XmlRootAttribute root = new XmlRootAttribute("DirectedGraph");
            root.Namespace = "http://schemas.microsoft.com/vs/2009/dgml";
            XmlSerializer serializer = new XmlSerializer(typeof(Graph), root);
            XmlWriterSettings settings = new XmlWriterSettings();
            settings.Indent = true;
            XmlWriter xmlWriter = XmlWriter.Create(xmlpath, settings);
            serializer.Serialize(xmlWriter, g);
        }
    }
}
