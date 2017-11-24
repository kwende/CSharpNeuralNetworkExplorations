using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public class DropoutLayerOptions
    {
        public int[] DropoutLayerIndices { get; private set; }

        public DropoutLayerOptions(params int[] layerIndices)
        {
            DropoutLayerIndices = layerIndices;
        }
    }
}
