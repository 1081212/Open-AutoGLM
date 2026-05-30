from phone_agent.device_factory import get_device_factory


def main():
    device_factory = get_device_factory()
    devices = device_factory.list_devices()
    print(devices)

if __name__ == "__main__":
    main()
