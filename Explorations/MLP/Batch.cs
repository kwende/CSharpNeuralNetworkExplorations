using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    public class Batch
    {
        public Batch()
        {
            Data = new List<TrainingData>();
        }

        public int Size
        {
            get
            {
                return Data.Count;
            }
        }

        public List<TrainingData> Data { get; set; }

        public static Batch[] CreateBatches(List<TrainingData> allData, int batchSize, Random rand)
        {
            int numberOfBatches = (int)System.Math.Ceiling(allData.Count / (1.0f * batchSize));

            // copy the list so as to not muck with the list passed in. 
            List<TrainingData> copyOfData = new List<TrainingData>(allData);
            // randomly sort the list. 
            copyOfData = copyOfData.OrderBy(n => rand.Next()).ToList();
            // return batch size. 
            Batch[] ret = new Batch[numberOfBatches];

            // create numberOfBatches batches. 
            int i = 0;
            for (int c = 0; c < numberOfBatches; c++)
            {
                Batch newBatch = new Batch();

                // randomly fill the batch. 
                for (int k = 0; k < batchSize && i < allData.Count; k++, i++)
                {
                    newBatch.Data.Add(copyOfData[i]);
                }

                ret[c] = newBatch;
            }

            return ret;
        }
    }
}
