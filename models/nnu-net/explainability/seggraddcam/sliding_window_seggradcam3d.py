"""
sliding_window_seggradcam3d.py

3D Segmentation GradCAM for nnU-Net v2.

Features:

- Uses nnU-Net sliding window geometry
- Uses Gaussian blending
- Works on 3D CT volumes
- Produces full-volume CAM
- Uses segmentation-aware target
- Supports GGO segmentation


Input:

preprocessed CT tensor

shape:
(1,1,D,H,W)


GT mask:

(D,H,W)


Output:

heatmap:

(D,H,W)

"""



import numpy as np

import torch

import torch.nn.functional as F



try:

    from nnunetv2.inference.sliding_window_prediction import (
        compute_gaussian
    )

except:

    compute_gaussian = None



from .hooks import GradCAMHooks


from .target import segmentation_target






class SlidingWindowSegGradCAM3D:



    def __init__(
            self,
            predictor,
            target_layer=None
    ):


        self.predictor = predictor

        self.network = predictor.network


        self.device = predictor.device



        # -----------------------------
        # nnU-Net settings
        # -----------------------------


        self.patch_size = tuple(
            predictor.configuration_manager.patch_size
        )


        self.step_size = (
            predictor.tile_step_size
        )


        self.use_gaussian = (
            predictor.use_gaussian
        )



        # -----------------------------
        # Disable deep supervision
        # -----------------------------


        self.network.decoder.deep_supervision = False



        # -----------------------------
        # Select CAM layer
        # -----------------------------


        if target_layer is None:


            target_layer = (
                self.network
                .decoder
                .stages[-1]
            )



        self.hooks = GradCAMHooks(
            target_layer
        )



        self.gaussian_cache = {}





    # ==================================================
    # Gaussian map
    # ==================================================


    def get_gaussian(
            self,
            patch_size
    ):


        if patch_size in self.gaussian_cache:

            return self.gaussian_cache[patch_size]



        if compute_gaussian is not None:


            gaussian = compute_gaussian(
                patch_size,
                sigma_scale=1/8
            )


            gaussian = torch.from_numpy(
                gaussian
            ).float()


        else:


            gaussian = torch.ones(
                patch_size
            )



        gaussian = gaussian.to(
            self.device
        )


        self.gaussian_cache[patch_size] = gaussian



        return gaussian






    # ==================================================
    # Sliding window positions
    # ==================================================


    def get_patch_locations(
            self,
            volume_shape
    ):


        locations=[]


        D,H,W = volume_shape


        pd,ph,pw = self.patch_size



        for d in range(
            0,
            max(D-pd+1,1),
            int(pd*self.step_size)
        ):


            for h in range(
                0,
                max(H-ph+1,1),
                int(ph*self.step_size)
            ):


                for w in range(
                    0,
                    max(W-pw+1,1),
                    int(pw*self.step_size)
                ):


                    locations.append(
                        (
                            d,h,w
                        )
                    )


        return locations







    # ==================================================
    # GradCAM computation
    # ==================================================


    def compute_patch_cam(
            self,
            patch,
            mask,
            target_class
    ):


        self.network.zero_grad()



        logits = self.network(
            patch
        )



        if isinstance(
            logits,
            (list,tuple)
        ):

            logits = logits[0]



        score = segmentation_target(

            logits,

            mask,

            target_class

        )



        score.backward()



        activations = (
            self.hooks.activations
        )


        gradients = (
            self.hooks.gradients
        )



        weights = gradients.mean(
            dim=(2,3,4),
            keepdim=True
        )



        cam = (
            weights *
            activations
        ).sum(
            dim=1,
            keepdim=True
        )



        cam = F.relu(
            cam
        )



        cam = F.interpolate(

            cam,

            size=self.patch_size,

            mode="trilinear",

            align_corners=False

        )



        cam = cam.squeeze()



        self.hooks.clear()



        return cam.detach()






    # ==================================================
    # Main function
    # ==================================================


    def __call__(

            self,

            input_volume,

            gt_mask,

            target_class=1

    ):



        input_volume = (
            input_volume.to(
                self.device
            )
        )



        gt_mask = (
            gt_mask.to(
                self.device
            )
        )



        volume_shape = (
            input_volume.shape[2:]
        )



        cam_volume = torch.zeros(
            volume_shape,
            device=self.device
        )


        weight_volume = torch.zeros(
            volume_shape,
            device=self.device
        )



        gaussian = self.get_gaussian(
            self.patch_size
        )



        locations = self.get_patch_locations(
            volume_shape
        )



        print(
            "CAM patches:",
            len(locations)
        )



        for d,h,w in locations:



            patch = input_volume[

                :,

                :,

                d:d+self.patch_size[0],

                h:h+self.patch_size[1],

                w:w+self.patch_size[2]

            ].clone().requires_grad_(True)



            mask_patch = gt_mask[

                d:d+self.patch_size[0],

                h:h+self.patch_size[1],

                w:w+self.patch_size[2]

            ]



            # skip empty patches

            if mask_patch.sum()==0:

                continue



            cam = self.compute_patch_cam(

                patch,

                mask_patch,

                target_class

            )



            cam_volume[

                d:d+self.patch_size[0],

                h:h+self.patch_size[1],

                w:w+self.patch_size[2]

            ] += cam * gaussian



            weight_volume[

                d:d+self.patch_size[0],

                h:h+self.patch_size[1],

                w:w+self.patch_size[2]

            ] += gaussian





        heatmap = (

            cam_volume /

            (weight_volume+1e-8)

        )



        # normalize 0-1


        heatmap = (

            heatmap -

            heatmap.min()

        ) / (

            heatmap.max() -

            heatmap.min()

            +

            1e-8

        )



        return heatmap.cpu().numpy()





    def remove_hooks(self):

        self.hooks.remove()