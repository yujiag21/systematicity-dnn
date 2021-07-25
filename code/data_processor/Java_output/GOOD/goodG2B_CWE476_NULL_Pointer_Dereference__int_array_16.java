    private void goodG2B() throws Throwable
    {
        int [] data;

        while (true)
        {
            /* FIX: hardcode data to non-null */
            data = new int[5];
            break;
        }

        while (true)
        {
            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length);
            break;
        }

    }
