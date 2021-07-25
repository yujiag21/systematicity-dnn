    private void goodG2B() throws Throwable
    {
        String data;

        while (true)
        {
            /* FIX: hardcode data to non-null */
            data = "This is not null";
            break;
        }

        while (true)
        {
            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length());
            break;
        }

    }
