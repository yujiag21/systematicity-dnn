    private void goodG2B() throws Throwable
    {
        Integer data;

        while (true)
        {
            /* FIX: hardcode data to non-null */
            data = Integer.valueOf(5);
            break;
        }

        while (true)
        {
            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.toString());
            break;
        }

    }
