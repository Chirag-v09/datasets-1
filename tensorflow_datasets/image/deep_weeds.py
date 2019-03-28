"""Dataset class for DeepWeeds dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import os

_URL = "https://nextcloud.qriscloud.org.au/index.php/s/a3KxPawpqkiorST/download"

_DESCRIPTION = ("""The DeepWeeds dataset consists of 17,509 images capturing eight different weed species native to Australia """ 
"""in situ with neighbouring flora.The selected weed species are local to pastoral grasslands across the state of Queensland."""
"""The images were collected from weed infestations at the following sites across Queensland: "Black River", "Charters Towers", """
""" "Cluden", "Douglas", "Hervey Range", "Kelso", "McKinlay" and "Paluma".""")

_NAMES = ["Chinee apple", "Snake weed", "Lantana", "Prickly acacia", "Siam weed", "Parthenium", "Rubber vine", "Parkinsonia", "Negative"]

_CITATION = """\

 @article{DeepWeeds2019,
  author = {Alex Olsen and
    Dmitry A. Konovalov and
    Bronson Philippa and
    Peter Ridd and
    Jake C. Wood and
    Jamie Johns and
    Wesley Banks and
    Benjamin Girgenti and
    Owen Kenny and 
    James Whinney and
    Brendan Calvert and
    Mostafa {Rahimi Azghadi} and
    Ronald D. White},
  title = {{DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning}},
  journal = {Scientific Reports},
  year = 2019,
  number = 2058,
  month = 2,
  volume = 9,
  issue = 1,
  day = 14,
  url = "https://doi.org/10.1038/s41598-018-38343-3",
  doi = "10.1038/s41598-018-38343-3"
}

"""

class DeepWeeds(tfds.core.GeneratorBasedBuilder):
  """DeepWeeds Image Dataset Class"""

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
        """Define Dataset Info"""

        return tfds.core.DatasetInfo(
            builder=self,
    
            description=(_DESCRIPTION),
            
            features=tfds.features.FeaturesDict({
                "image":tfds.features.Image(),
                "label": tfds.features.ClassLabel(names=_NAMES),
            }),

            supervised_keys=("image", "label"),
            
            urls=[_URL],
            
            citation=_CITATION
        )

  def _split_generators(self, dl_manager):
        """Define Splits"""
        path = dl_manager.download_and_extract(_URL)

        return [
            tfds.core.SplitGenerator(
                name="train",
                num_shards=10,
                gen_kwargs={
                    "data_dir_path": path,
                },
            ),
        ]

  def _generate_examples(self,data_dir_path):
      """Generate images and labels for splits"""
      
      for file_name in tf.io.gfile.listdir(data_dir_path):
          image = os.path.join(data_dir_path,file_name)
          label = _NAMES[int(file_name.split("-")[2].split(".")[0])]

          yield{
              "image":image,
              "label":label
          }
