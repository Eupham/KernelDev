import modal

volume = modal.Volume.from_name("kernel-dev-volume")
try:
    volume.delete()
    print("Volume 'kernel-dev-volume' deleted.")
except Exception as e:
    print(f"Could not delete volume: {e}")

modal.Volume.from_name("kernel-dev-volume", create_if_missing=True)
print("Volume 'kernel-dev-volume' re-created.")
