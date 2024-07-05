import pandas as pd

data1 = {
    "124012500223026_9_RH_WELD1": {
        "diff_pixel": {
            "ROI": 4435.0,
            "FILM": 77365.0,
            "MARK": 164585.0
        },
        "diff_iou": {
            "ROI": [
                0.9459446474000245
            ],
            "FILM": [
                0.9478513699862695
            ],
            "MARK": [
                0.8153265716881908
            ]
        }
    },
    "124012500565414_9_RH_WELD1": {
        "diff_pixel": {
            "ROI": 3357.0,
            "FILM": 73992.0,
            "MARK": 164617.0
        },
        "diff_iou": {
            "ROI": [
                0.9522420866210508
            ],
            "FILM": [
                0.9532504598163589
            ],
            "MARK": [
                0.8530265307996915
            ]
        }
    }
}

data2 = {
    "124012500223026_9_RH_WELD1": {
        "diff_pixel": {
            "ROI": 2645.0,
            "FILM": 76595.0,
            "MARK": 166833.0
        },
        "diff_iou": {
            "ROI": [
                0.961504948231604
            ],
            "FILM": [
                0.9775621769634649
            ],
            "MARK": [
                0.872769302540284
            ]
        }
    },
    "124012500565414_9_RH_WELD1": {
        "diff_pixel": {
            "ROI": 1763.0,
            "FILM": 69593.0,
            "MARK": 162906.0
        },
        "diff_iou": {
            "ROI": [
                0.9728958125791726
            ],
            "FILM": [
                0.982057866530164
            ],
            "MARK": [
                0.8849333080543932
            ]
        }
    }
}

comparison_data = []

for key1, val1 in data1.items():
    val2 = data2[key1]
    
    data1_diff_pixel = val1["diff_pixel"]
    data1_diff_iou = val1["diff_iou"]
    data2_diff_pixel = val2["diff_pixel"]
    data2_diff_iou = val2["diff_iou"]
    
    for region in data1_diff_pixel.keys():
        comparison_data.append({
            "Image": key1,
            "Region": region,
            "Data1_diff_pixel": data1_diff_pixel[region],
            "Data2_diff_pixel": data2_diff_pixel[region],
            "Data1_diff_iou": data1_diff_iou[region][0],
            "Data2_diff_iou": data2_diff_iou[region][0],
        })

# Convert to DataFrame
df_comparison = pd.DataFrame(comparison_data)

# Print the comparison table
print(df_comparison)