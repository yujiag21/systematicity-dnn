    private void goodG2B() throws Throwable
    {
        StringBuilder data;

        while (true)
        {
            /* FIX: hardcode data to non-null */
            data = new StringBuilder();
            break;
        }

        while (true)
        {
            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length());
            break;
        }

    }
