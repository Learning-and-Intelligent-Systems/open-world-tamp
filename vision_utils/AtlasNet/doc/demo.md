# Demo



Simply run `train.py –demo` to download the pre-trained models and run on an example airplane image. The output will be saved in PLY format as `./doc/pictures/plane_input_demoAtlasnetReconstruction.ply`. You can view the output by downloading [MeshLab] (http://www.meshlab.net/).




### Trained models

Trained are automatically downloaded by `train.py --demo ...`.

You can also explicitly download them with:

```
chmod +x training/download_trained_models.sh
./training/download_trained_models.sh
```


All training options can be recovered in `{dir_name}/options.txt`.

* `./training/trained_models/atlasnet_autoencoder_25_squares/network.pth` [Default]

* `./training/trained_models/atlasnet_autoencoder_1_sphere/network.pth` 

* `./training/trained_models/atlasnet_singleview_25_squares/network.pth` [Default]

* `./training/trained_models/atlasnet_singleview_1_sphere/network.pth` 

  
### Usage

```python train.py --demo --demo_input_path YOUR_IMAGE_or_OBJ_PATH --reload_model_path YOUR_MODEL_PTH_PATH ```

```
This function takes an image or pointcloud path as input and save the mesh infered by Atlasnet
Extension supported are `ply` `npy` `obg` and `png`
--demo_input_path input file e.g. image.png or object.ply 
--reload_model_path trained model path (see below for pretrained models) 
:return: path to the generated mesh
```



To generate the example below, simple run `python train.py --demo`. It will (1) default to the 2D plane image as input, (2) download a trained single-view Altasnet with 25 square primitives, (3) run the image through the network and (4) save the generated 3D plane in `doc/pictures/`.



![input](./pictures/2D3D.png)





You can use our  [Meshlab Visualization Trick](./doc/meshlab.md) to have nicer visualization of the generated mesh in Meshlab.
