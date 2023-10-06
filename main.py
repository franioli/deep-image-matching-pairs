def run_tests():
    try:
        import deep_image_matcher
    except ImportError as e:
        raise ImportError(e)

    # retcode = pytest.main()
    # sys.exit(retcode)


if __name__ == "__main__":
    run_tests()
